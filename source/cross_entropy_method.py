import copy
from pathlib import Path
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt

from source.envs import Observer, plot_trajectories
from source.models import model_factory
from source.random_policy import RandomPolicy
from source.training import Trainer

logger = logging.getLogger(__name__)


class CEMPlanner(object):
    def __init__(self,
                 env: object,
                 observer: Observer,
                 horizon: int = 20,
                 action_repeat: int = 1,
                 seed: int = 5,
                 receding_horizon: int = 25,
                 population: int = 300,
                 selection: int = 20,
                 iterations: int = 5,
                 use_oracle: bool = False,
                 device: str = "cuda",
                 model_class: str = "LinearModel",
                 model_path: str = "models/{}_model_{}.tar",
                 directory: str = "out",
                 model_fit_frequency: int = 200,
                 test: bool = False,
                 **kwargs: dict) -> None:
        self.seed = seed
        self.env = env
        self.observer = observer
        self.horizon = horizon
        self.action_repeat = action_repeat
        self.receding_horizon = receding_horizon
        self.population = population
        self.selection = selection
        self.iterations = iterations
        self.use_oracle = use_oracle
        self.device = device
        self.model_class = model_class
        self.model_path = model_path 
        self.directory = Path(directory)
        self.model_fit_frequency = model_fit_frequency
        self.test = test or self.use_oracle
        self.kwargs = kwargs
        self.dynamics_model = None
        self.trainer = None
        self.planned_actions = []
        self.history = []
        self.steps = 0
        self.gamma = 0.99
                
        if not self.use_oracle:
            
            self.load_dynamics(model_class=self.model_class,
                               model_path=self.model_path,
                               **self.kwargs)

            self.trainer = Trainer(self.dynamics_model,
                                   device=self.device,
                                   directory=self.directory,
                                   model_path=self.model_path,
                                   seed=self.seed,
                                   **self.kwargs)
        self.exploration_policy = RandomPolicy(env, **kwargs)

    @property
    def name(self):
        return self.dynamics_model.__class__.__name__

    def load_dynamics(self, model_class, model_path, **kwargs):
        kwargs["action_size"] = self.env.action_space.shape[0]

        kwargs["state_size"] = self.observer.observe_array(self.env).size
        self.dynamics_model = model_factory(model_class, kwargs)
        if self.test:
            path = self.directory / model_path.format(self.dynamics_model.__class__.__name__, self.seed)
            try:
                self.dynamics_model.load_state_dict(torch.load(path))
                print(f"Loaded {model_class} from {path}.")
            except:
                
                if "Perfect" in model_class:
                    pass
                else:
                    print('cannot load' , model_class)
                    
            self.dynamics_model.eval()
        self.dynamics_model.to(self.device)

        return self

    def fit_dynamics(self):
        # Reset initial model and optimizer
        self.load_dynamics(model_class=self.model_class,
                           model_path=self.model_path,
                           **self.kwargs)
        self.trainer = Trainer(self.dynamics_model,
                               device=self.device,
                               directory=self.directory,
                               model_path=self.model_path,
                               seed=self.seed,
                               **self.kwargs)
        # Train
        self.trainer.train(df=self.observer.dataframe())
        self.trainer.save_model()
        self.trainer.plot_losses()
        plt.close()

    def reward_model(self, states, actions, gamma=None):
        """
        :param Tensor states: a batch of states. shape: [?time, batch_size, state_size].
        :param float gamma: a discount factor
        """
        if  self.env.classenv_name == "cartpolesw":

            x = states[...,0]
            th = states[...,2]
            xdot = states[...,1]
            thdot = states[...,3]       

            rewards = torch.exp(-((x + 0.5*torch.sin(th))** 2 + (0.5*torch.cos(th))**2 ))

        elif  self.env.classenv_name == "cartpole":
            x = states[..., 0]
            theta = states[..., 2]
            theta_threshold_radians = 12 * 2 * torch.pi / 360
            x_threshold = 2.4
            done = (
    	        (x < -x_threshold)
    	        | (x > x_threshold)
    	        | (theta < -theta_threshold_radians)
    	        | (theta > theta_threshold_radians)
    	        )
            rewards = torch.tensor( done.logical_not().float())

        elif  self.env.classenv_name == "ctacrobot":
            s = states
            done = (
    	    (-s[...,0] - (s[...,0]*s[...,2] - s[...,1]*s[...,3]) > 1.0)
    	    )        
            rewards = torch.tensor( - done.logical_not().float()) 
    
        elif  self.env.classenv_name == "acrobot":
            rewards = (- torch.cos(states[..., 0]) - torch.cos(states[..., 0] + states[..., 1] )) / 2

        elif self.env.classenv_name == "pendulum" or self.env.classenv_name == "pendulumsw"  :
            th = states[...,0]
            thdot = states[...,1]
            rewards = -(angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (actions.squeeze()**2))



        if gamma:
            time = torch.arange(rewards.shape[0], dtype=torch.float).unsqueeze(-1).expand(rewards.shape)
            rewards *= torch.pow(gamma, time)
            
        
        return rewards

    def predict_trajectory(self, state, actions):
        """
        Predict trajectories from sequences of actions by forwarding a model

        :param state: [ batch x state size ]
        :param actions: [ horizon x batch x action size ]
        :return: [ horizon x batch x state size]
        """
        actions = torch.repeat_interleave(actions, self.action_repeat, dim=0)

        states = self.dynamics_model.integrate(state, actions)
        return states[::self.action_repeat, ...]

    def predict_trajectory_oracle(self, env, initial_state, actions):
        """
        Predict trajectories from sequences of actions by stepping the true environment
        :param env: an environment to step
        :param initial_state: initial qpos, qvel, qacc of the environment, corresponding to the current state
        :param actions: [ actions count x batch x action size ]
        :return: [(horizon = actions*repeat) x batch x state size]
        """
        all_trajs = []
        actions = actions.cpu().numpy()
        for env_i in range(actions.shape[1]):
            self.restore_env_state(env, initial_state)
            trajectory = [self.observer.observe_array(env)]
            for t in range(actions.shape[0]):
                action = actions[t, env_i]
                for _ in range(self.action_repeat):
                    obs, reward, done, info = env.step(action)
                trajectory.append(self.observer.observe_array(env))
            all_trajs.append(torch.tensor(trajectory, dtype=torch.float))
        return torch.stack(all_trajs, dim=1)

    def plan(self, state):
        """
        Cross Entropy Method.

        :return: dict of the planned trajectories (times, states and actions)
        """
        action_space = self.env.action_space
        action_mean = torch.zeros(self.horizon, 1, action_space.shape[0], device=self.device)
        action_std = torch.ones(self.horizon, 1, action_space.shape[0], device=self.device) * action_space.high.max()
        env_state, env_copy = None, None
        if self.use_oracle:
            
            env_copy = self.env
            env_state = self.get_env_state(self.env)

        else:
            state = torch.tensor(self.observer.observe_array(self.env, state), dtype=torch.float).to(self.device)
            state = state.expand(self.population, -1)
        with torch.no_grad():
            for _ in range(self.iterations):
                # 1. Draw sample sequences of actions from a normal distribution
                actions = action_mean + action_std * torch.randn(self.horizon, self.population, action_space.shape[0], device=self.device)
                actions = torch.clamp(actions, min=action_space.low.min(), max=action_space.high.max())
                
                # 2. Unroll trajectories
                if self.use_oracle:
                    states = self.predict_trajectory_oracle(env_copy, env_state, actions)
                else:

                    states = self.predict_trajectory(state, actions)
                # 3. Fit the distribution to the top-k performing sequences
                returns = self.reward_model(states, actions).sum(dim=0)
                _, best = returns.topk(self.selection, largest=True, sorted=False)
                states = states[:-1, :, :]  # Remove last predicted state, for which we have no action
                best_actions = actions[:, best, :]
                action_mean = best_actions.mean(dim=1, keepdim=True)
                action_std = best_actions.std(dim=1, unbiased=False, keepdim=True)
        times = self.observer.time + np.arange(states.shape[0]) * self.observer.dt(self.env) * self.action_repeat
        if self.use_oracle:
            self.restore_env_state(env_copy, env_state)


        return {
            "states": states[:, best, :].cpu().numpy(),
            "actions": best_actions.cpu().numpy(),
            "time": times
        }

    @staticmethod
    def get_env_state(env):
        if hasattr(env, "dmcenv"):
            return (env.unwrapped.dmcenv.physics.data.qpos.copy(),
                     env.unwrapped.dmcenv.physics.data.qvel.copy(),
                     env.unwrapped.dmcenv.physics.data.qacc_warmstart.copy())
        elif hasattr(env, "scene"):
            return env.unwrapped.scene._p.saveState(), env._elapsed_steps
        else:
            return env.unwrapped.state

    @staticmethod
    def restore_env_state(env, state):
        if hasattr(env, "dmcenv"):
            with env.unwrapped.dmcenv.physics.reset_context():
                    qpos, qvel, qacc_ws = state
                    env.unwrapped.dmcenv.physics.data.qpos[:] = qpos
                    env.unwrapped.dmcenv.physics.data.qvel[:] = qvel
                    env.unwrapped.dmcenv.physics.data.qacc_warmstart[:] = qacc_ws
        elif hasattr(env, "scene"):
            bul_state, steps = state
            env.unwrapped.scene._p.restoreState(bul_state)
            env._elapsed_steps = steps
        else:
            env.unwrapped.state = state

    def step(self):
        if not self.test:
            self.exploration_policy.step()
            self.steps += 1
            
            if self.steps % self.model_fit_frequency == 0:

                self.fit_dynamics()



    def reset(self):
        self.planned_actions = []
        self.history = []

    def act(self, state):
        self.step()
        if not self.planned_actions:
            trajectories = self.plan(state)
            self.planned_actions = trajectories["actions"].mean(axis=1)
            self.planned_actions = np.repeat(self.planned_actions, self.action_repeat, axis=0)[:-self.action_repeat, :].tolist()
            self.planned_actions = self.planned_actions[:self.receding_horizon]
            self.history.append(trajectories)
        best_action = np.array(self.planned_actions.pop(0))
        action = best_action if self.test else self.exploration_policy.act(best_action)
        return action

    def plot(self, episode_dataframe, directory=None,filename="trajectories_{}.png"):
        axes = None
        directory = Path(directory or self.directory)

        # True trajectory
        df = episode_dataframe
        df_states = df.filter(like='state').to_numpy()
        df_actions = df.filter(like='action').to_numpy()
        axes = plot_trajectories(df["time"], df_states, df_actions, self.env.action_space,
                                 plot_args={"linestyle": '-', "linewidth": 2}, axes=axes)

        # Planned trajectories

        traj_data = {key: np.concatenate([data[key] for data in self.history], axis=0) for key in self.history[0].keys()}
        axes = plot_trajectories(traj_data["time"], traj_data["states"], traj_data["actions"], self.env.action_space,
                                 plot_args={"linestyle": '--', "alpha": 0.5}, axes=axes)

        directory.mkdir(parents=True, exist_ok=True)
        plt.savefig(directory / filename.format(self.name))
        
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi