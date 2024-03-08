import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(THIS_DIR)
sys.path.append(PARENT_DIR)


from pathlib import Path

import torch
import tqdm
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from source.utils import str2bool, display_video

sns.set_style("whitegrid")

from source.agents import agent_factory
from source.envs import env_factory, plot_trajectories
import argparse


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, help="the environment", default="acrobot")
    parser.add_argument("--agent", type=str, help="the agent used for planning (class name)", default="PHIHP")
    parser.add_argument("--agent_name", type=str, help="the agent used for planning (class name)", default=None)    
    parser.add_argument("--seed", type=int, help="the seed in which the model was trained", default=None)
    # Model arguments
    parser.add_argument("--model_class", type=str, help="the model class")
    parser.add_argument("--device", type=str, help="the device to load the model on", default="cuda")

    # PhIHP planner arguments
    parser.add_argument("--PHIHP_role", type=str2bool, help="Use PHIHP or Im_td3", nargs='?', const=True, default=True,)
    parser.add_argument("--use_Q", type=str2bool, help="use Q value with reward for trajectory evaluation or not", nargs='?', const=True, default=True,)
    parser.add_argument("--alpha", type=float, help="regularization", default=0.2)    
    parser.add_argument("--horizon", type=int, help="planning horizon", default=20)
    parser.add_argument("--action_repeat", type=int, help="action repeat", default=1)
    parser.add_argument("--receding_horizon", type=int, help="receding horizon for MPC", default=25)
    parser.add_argument("--population", type=int, help="random population size for CEM", default=50)
    parser.add_argument("--pi_population", type=int, help="policy population size from TD3", default=50)    
    parser.add_argument("--selection", type=int, help="selected population size for CEM", default=10)
    parser.add_argument("--iterations", type=int, help="number of iterations in CEM", default=3)



    # Evaluation arguments
    parser.add_argument("--episodes", type=int, help="number of episodes", default=10)
    parser.add_argument("--model_path", type=str, help="model path", default="out")
    parser.add_argument("--rl_path", type=str, help="policy path", default="out")
    parser.add_argument("--directory", type=str, help="run directory", default="out")



    args = parser.parse_args(raw_args)


    env_test, observer_test = env_factory(args.environment)

    seed = args.seed

    agent = agent_factory(env_test, observer_test, args.agent, **vars(args))


    rl_path = args.rl_path

    agent.load_dynamics(agent.model_class,args.model_path )


    PHIHP_reward = display_PHIHP( agent, env_test, int(args.episodes), rl_path, seed, Path(args.directory))
    print(PHIHP_reward)



    plt.close()

def display_PHIHP(agent, env_test, episodes, rl_path, seed, dir):


        frames = []


        agent.rl.load(rl_path + f"episode{episodes}_seed{seed}")
        episode_rewards = []

        agent.test = True
        agent.env = env_test
        state, _ = agent.env.reset()
        agent.reset()
        done, terminated = False, False
        ep_reward = 0
        while not (done or terminated):
            action = agent.act(state)
            next_state, reward, terminated, done, info = env_test.step([action])
            state = next_state
            ep_reward += reward
            try:    
                pixels = env_test.render()
                frames.append(pixels)
            except :
                pass
        episode_rewards.append(ep_reward)
        
        display_video(frames, path=f"{dir}/episode_{episodes}_.mp4", framerate=60)
        

        return ep_reward











if __name__ == '__main__':
    main()


