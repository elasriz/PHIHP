import copy
from pathlib import Path
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt

from source.envs import Observer, plot_trajectories
from source.models import model_factory

from source.TD3 import TD3, ExplorationNoise
from source.envs import env_factory

import os

logger = logging.getLogger(__name__)


class PHIHP(object):
    def __init__(self,
                 env: object,
                 observer: Observer,     
                 horizon: int = 20,
                 seed: int = 5,
                 action_repeat: int = 1,
                 receding_horizon: int = 25,
                 population: int = 300,
                 pi_population: int = 20,
                 selection: int = 20,
                 iterations: int = 5,
                 PHIHP_role: bool = False,
                 use_Q: bool = False,
                 alpha: float = 0.1,
                 device: str = "cuda",
                 model_class: str = "LinearModel",
                 model_path: str = "models/{}_model_{}.tar",
                 directory: str = "out",
                 model_fit_frequency: int = 200,
                 test: bool = True,
                 **kwargs: dict) -> None:
        

        self.test = test
        self.seed = seed
        


        self.env = env


        self.mbrl_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

        self.observer = observer
        self.horizon = horizon
        self.action_repeat = action_repeat
        self.receding_horizon = receding_horizon
        self.population = population
        self.pi_population = pi_population 
        self.selection = selection
        self.iterations = iterations
        self.PHIHP_role = PHIHP_role
        self.use_Q = use_Q
        self.alpha = alpha

        self.device = device
        self.model_class = model_class
        self.model_path = model_path
        self.directory = Path(directory)
        self.model_fit_frequency = model_fit_frequency
        
        self.kwargs = kwargs
        self.dynamics_model = None
        self.planned_actions = []
        self.history = []
        self.gamma = 0.99
   



        if self.env.unwrapped.classenv_name == "pendulum" or self.env.unwrapped.classenv_name == "pendulumsw":
            state_dim = self.env.observation_space.shape[0] + 1   
           
        elif self.env.unwrapped.classenv_name == "cartpolesw":
            state_dim = self.env.observation_space.shape[0] + 1 
        elif  self.env.unwrapped.classenv_name == "acrobotsw" or self.env.unwrapped.classenv_name == "ctacrobot":
            state_dim = self.env.observation_space.shape[0] + 2
        else:
            state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]     

        self.rl = TD3(state_dim=state_dim, action_dim=action_dim, actor_hidden_dim=[300,300], critic_hidden_dim=[200,200,200], tau=0.005, policy_noise=0.1) 
        self.rl.exploration_noise = ExplorationNoise(action_dim=action_dim)  


        

    @property
    def name(self):
        return self.dynamics_model.__class__.__name__

    def load_dynamics(self, model_class, model_path, **kwargs):
        kwargs["action_size"] = self.env.action_space.shape[0]
        kwargs["state_size"] = self.env.observation_space.shape[0]
        self.dynamics_model = model_factory(model_class, kwargs)

        path = os.path.join(self.mbrl_directory, model_path.format(self.dynamics_model.__class__.__name__, self.seed)) 
        self.dynamics_model.load_state_dict(torch.load(path))
        print(f"Loaded {model_class} from {path}.")
        self.dynamics_model.eval()
        self.dynamics_model.to(self.device)
        return self

    def reset(self):
        self.planned_actions = []
        self.history = []

    def reward_model(self, states, actions, gamma=None):
        """

        """

        if  self.env.unwrapped.classenv_name == "cartpolesw":

            x = states[...,0]
            th = states[...,2]
            xdot = states[...,1]
            thdot = states[...,3]       

            rewards = torch.exp(-((x + 0.5*torch.sin(th))** 2 + (0.5*torch.cos(th))**2 ))

        elif  self.env.unwrapped.classenv_name == "cartpole":
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
            rewards = done.logical_not().float().clone().detach()

        elif  self.env.unwrapped.classenv_name == "ctacrobot":
            s = states
            done = (
    	    (-s[...,0] - (s[...,0]*s[...,2] - s[...,1]*s[...,3]) > 1.0)
    	    )        
            rewards = -done.logical_not().float().clone().detach()
    
        elif  self.env.unwrapped.classenv_name == "acrobotsw":
            rewards = (- torch.cos(states[..., 0]) - torch.cos(states[..., 0] + states[..., 1] )) / 2

        elif self.env.unwrapped.classenv_name == "pendulum" or self.env.unwrapped.classenv_name == "pendulumsw"  :
            th = states[...,0]
            thdot = states[...,1]
            rewards = -(angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (actions.squeeze()**2))



        if gamma:
            time = torch.arange(rewards.shape[0], dtype=torch.float).unsqueeze(-1).expand(rewards.shape).to(self.device)
            rewards *= torch.pow(gamma, time)


        if self.use_Q:
            
            states_new = states[-1].to(dtype=torch.float32, device=self.device)  
            action =  self.rl.actor(self._get_obs(states_new))            
            action_ = action.clone().detach().to(dtype=torch.float32, device=self.device)
            q1 = self.rl.critic1(self._get_obs(states_new), action_)
            q2 = self.rl.critic2(self._get_obs(states_new), action_)
            Q_value = self.alpha * (gamma**self.horizon) * torch.min(q1,q2).reshape(1,self.population + self.pi_population)
            rewards = torch.cat([rewards, Q_value], dim=0)

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



    def _get_obs(self, state):

        if self.env.unwrapped.classenv_name == "pendulum" or self.env.unwrapped.classenv_name == "pendulumsw":
            aug_state = torch.zeros(state.shape[0],state.shape[1]+1, device=self.device)
            aug_state[:,0] = torch.cos(state[:,0])
            aug_state[:,1] = torch.sin(state[:,0])
            aug_state[:,2] = state[:,1]
            return aug_state

        elif  self.env.unwrapped.classenv_name == "acrobotsw" or self.env.unwrapped.classenv_name == "ctacrobot":


            aug_state = torch.zeros(state.shape[0],state.shape[1]+2, device=self.device)
            aug_state[:,0] = torch.cos(state[:,0])
            aug_state[:,1] = torch.sin(state[:,0])
            aug_state[:,2] = torch.cos(state[:,1])
            aug_state[:,3] = torch.sin(state[:,1])      
            aug_state[:,4] = state[:,2]
            aug_state[:,5] = state[:,3]     
            return aug_state

        elif self.env.unwrapped.classenv_name == "cartpolesw":
            aug_state = torch.zeros(state.shape[0],state.shape[1]+1, device=self.device)
            aug_state[:,0] = state[:,0]
            aug_state[:,1] = state[:,1]
            aug_state[:,2] = torch.cos(state[:,2])
            aug_state[:,3] = torch.sin(state[:,2])      
            aug_state[:,4] = state[:,3]
            return aug_state 
        
        else:
            return state


  
    
    def plan_free(self, state):

        if not self.test and len(self.rl.replay_buffer) < 5000:
                return [self.env.action_space.sample()]
        
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state_pi = self._get_obs(state.unsqueeze(0))
        return self.rl.select_action_TD(state_pi)
    
    def plan(self, state):
        """
        Cross Entropy Method.

        :return: dict of the planned trajectories (times, states and actions)
        """
        action_space = self.env.action_space

        action_mean = torch.zeros(self.horizon, 1, action_space.shape[0], device=self.device)
        action_std = torch.ones(self.horizon, 1, action_space.shape[0], device=self.device) * action_space.high.max()


        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state_pi = state.unsqueeze(0) 

            
        state = state.expand(self.population + self.pi_population, -1) 
        
        # 1. Draw sample sequences of actions from RL policy 
        pi_actions = torch.empty(self.horizon, self.pi_population, action_space.shape[0], device=self.device)
        
        z = state_pi.repeat(self.pi_population, 1)
      

        with torch.no_grad():

            for t in range(self.horizon):

                pi_actions[t] = self.rl.sample_pi_actions(self._get_obs(z)) #torch.from_numpy(self.rl.select_action(z))
                z = self.dynamics_model.integrate(z, pi_actions[t].view(1,self.pi_population,action_space.shape[0]))[0]


            # 1.bis Draw sample sequences of actions from a normal distribution
            actions = action_mean + action_std * torch.randn(self.horizon, self.population, action_space.shape[0], device=self.device)
            actions = torch.clamp(actions, min=action_space.low.min(), max=action_space.high.max())

            actions = torch.cat([pi_actions, actions], dim=1)


            for _ in range(self.iterations):
                
                # 2. Unroll trajectories
                states = self.predict_trajectory(state, actions)
                # 3. Fit the distribution to the top-k performing sequences
                returns = self.reward_model(states, actions,self.gamma).sum(dim=0)
                _, best = returns.topk(self.selection, largest=True, sorted=False)
                states = states[:-1, :, :]  # Remove last predicted state, for which we have no action
                best_actions = actions[:, best, :]
                action_mean = best_actions.mean(dim=1, keepdim=True)
                action_std = best_actions.std(dim=1, unbiased=False, keepdim=True)
                actions = action_mean + action_std * torch.randn(self.horizon, self.population + self.pi_population, action_space.shape[0], device=self.device)
                actions = torch.clamp(actions, min=action_space.low.min(), max=action_space.high.max())

        times = self.observer.time + np.arange(states.shape[0]) * self.observer.dt(self.env) * self.action_repeat


        return {
            "states": states[:, best, :].cpu().numpy(),
            "actions": best_actions.cpu().numpy(),
            "time": times
        }


        
    def act(self, state):

        if not (self.test and self.PHIHP_role):
            action = self.plan_free(state)
            return action[0]

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