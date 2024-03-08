from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import gymnasium as gym


random_state = np.random.RandomState()


def env_factory(env_name: str,model_class=None, seed=None, dir="out") -> Tuple:
     
    if env_name == "pendulum":
        import envts.pendulum
        env = gym.make("FrPendulum")        
        observer = PendulumObserver()

    elif env_name == "im_pendulum":
        import envts.pendulum_fake
        env = gym.make("im_pendulum", model_class=model_class, model_seed=seed, directory=dir)        
        observer = PendulumObserver()

    elif env_name == "pendulumsw":
        import envts.pendulumsw
        env = gym.make("FrPendulumsw")        
        observer = PendulumObserver()

    elif env_name == "im_pendulumsw":
        import envts.pendulumsw_fake
        env = gym.make("im_pendulumsw", model_class=model_class, model_seed=seed, directory=dir)        
        observer = PendulumObserver()

    elif env_name == "ctcartpole":
        import envts.ctcartpole
        env = gym.make("FrictionCartpole")        
        observer = CartpoleObserver()  

    elif env_name == "im_cartpole":
        import envts.ctcartpole_fake
        env = gym.make("im_cartpole", model_class=model_class, model_seed=seed, directory=dir)        
        observer = CartpoleObserver()

    elif env_name == "ctcartpolesw":
        import envts.ctcartpolesw
        env = gym.make("FrictionCartpolesw")        
        observer = CartpoleObserver()  

    elif env_name == "im_cartpolesw":
        import envts.ctcartpolesw_fake
        env = gym.make("im_cartpolesw", model_class=model_class, model_seed=seed, directory=dir)        
        observer = CartpoleObserver()

    elif env_name == "ctacrobot":
        import envts.ctacrobot
        env = gym.make("ctacrobot")        
        observer = CartpoleObserver()  

    elif env_name == "im_ctacrobot":
        import envts.ctacrobot_fake
        env = gym.make("im_ctacrobot", model_class=model_class, model_seed=seed, directory=dir)        
        observer = CartpoleObserver()

    elif env_name == "ctacrobotsw":
        import envts.ctacrobotsw
        env = gym.make("ctacrobotsw")        
        observer = CartpoleObserver()  

    elif env_name == "im_ctacrobotsw":
        import envts.ctacrobotsw_fake
        env = gym.make("im_ctacrobotsw", model_class=model_class, model_seed=seed, directory=dir)        
        observer = CartpoleObserver()

    else:
        raise ValueError(f"Unknown env {env_name}")
    env.reset()
    return env, observer


class Observer(object):
    def __init__(self):
        self.episodes = [[]]
        self.time = 0

    def reset(self):
        self.time = 0
        self.episodes.append([])

    def dt(self, env):
        return env.unwrapped.dt

    def log(self, env, state, action):
        datapoint = self.observe(env, state)
        datapoint.update((f"action_{i}", action[i]) for i in range(action.size))
        datapoint[f"time"] = self.time
        self.time += self.dt(env)
        self.episodes[-1].append(datapoint)

    def dataframe(self):
        return pd.DataFrame(sum(self.episodes, []))

    def plot(self, path="out/dataset.png"):
        self.dataframe().plot(x="time")
        plt.savefig(path)
        plt.show()

    def save(self, path="out/data/dataset.csv"):
        print(f"Save data to {path}")
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        self.dataframe().to_csv(path)

    def append_to(self, path="out/data/dataset.csv"):
        try:
            df_load = pd.read_csv(path)
            print(f"Append data to {path}")
            df = pd.concat([df_load, self.dataframe()])
            df.to_csv(path)
        except FileNotFoundError:
            self.save(path)

    def observe(self, env, state=None):
        raise NotImplementedError()

    def observe_array(self, env, state=None):
        return np.array(list(self.observe(env, state).values()))


class PendulumObserver(Observer):
    def observe(self, env, state=None):
        if state is None:
            state = env.unwrapped.state
        return OrderedDict([
            ("state_angle", state[0]),
            ("state_angular_vel", state[1])
        ])


class AcrobotObserver(Observer):
    def observe(self, env, state=None):

        if state is None:
            state = env.unwrapped.state
            #state = env.reset()

        return OrderedDict([
            ("state_angle_1", state[0]),
            ("state_angle_2", state[1]),
            ("state_angular_vel_1", state[2]),
            ("state_angular_vel_2", state[3])
        ])
        
class CartpoleObserver(Observer):
    def observe(self, env, state=None):

        if state is None:
            state = env.unwrapped.state
            #state = env.reset()

        return OrderedDict([
            ("state_cart_position", state[0]),
            ("state_cart_velocity", state[1]),
            ("state_pole_angle", state[2]),
            ("state_pole_velocity", state[3])
        ])        
        
def plot_trajectories(time, states, actions, action_space, axes=None, plot_args={}):
    """
    Plot trajectories.
    H: horizon, S: state size, A: action size, B: number of trajectories

    :param time: shape H
    :param states: shape H x S or H x B x S
    :param actions: shape H x A or H x B x A
    :param action_space: action space
    :param axes: axes to plot into
    :param plot_args: arguments for plot
    :return: axes
    """
    # Mask time gaps
    gaps = np.where((np.diff(time) > 1.1*np.diff(time)) | (np.diff(time) < 0))
    
    intervals = [range(gaps[0][0]+1)] + [range(gaps[0][i]+1, gaps[0][i+1]+1) for i in range(gaps[0].size-1)] + \
                [range(gaps[0][-1]+1, time.size)] if gaps[0].size else [range(time.size)]

    if len(states.shape) == 3:
        states_mean = states.mean(axis=1)
        states_std = states.std(axis=1)
        actions_mean = actions.mean(axis=1)
        actions_std = actions.std(axis=1)
    else:
        states_mean, states_std = states, None
        actions_mean, actions_std = actions, None

    # Plot
    if axes is None:
        fig, axes = plt.subplots(states.shape[-1] + actions.shape[-1], 1, sharex=True)
    for tt in intervals:
        for i in range(states.shape[-1]):
            if states_std is not None:
                axes.flat[i].fill_between(
                    time[tt], states_mean[tt, i] + states_std[tt, i], states_mean[tt, i] - states_std[tt, i], **plot_args)
            else:
                axes.flat[i].plot(time[tt], states_mean[tt, i], **plot_args)
            axes.flat[i].set_xlabel("time")
            axes.flat[i].set_ylabel(f"state {i}")
        for i in range(actions.shape[-1]):
            if actions_std is not None:
                axes.flat[i + states.shape[-1]].fill_between(
                    time[tt], actions_mean[tt, i] + actions_std[tt, i], actions_mean[tt, i] - actions_std[tt, i], **plot_args)
            else:
                axes.flat[i + states.shape[-1]].plot(time[tt], actions_mean[tt, i], **plot_args)
            axes.flat[i + states.shape[-1]].set_xlabel("time")
            axes.flat[i + states.shape[-1]].set_ylabel(f"action {i}")
    [ax.grid() for ax in axes.flat]
    return axes
