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

from source.utils import str2bool, seed_all
from source.envs import env_factory
from source.TD3 import TD3, ExplorationNoise

import argparse
import time


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, help="the environment", default="im_acrobot")
    parser.add_argument("--seed", type=int, help="seed for the randomness", default=None)
    # Model arguments
    parser.add_argument("--model_class", type=str, help="the model class")
    parser.add_argument("--device", type=str, help="the device to load the model on", default="cuda")

    # Evaluation arguments
    parser.add_argument("--timesteps", type=int, help="number of timesteps for training", default=1000) 
    parser.add_argument("--starting_step", type=int, help=" timestep when to start training", default=10000)    
    parser.add_argument("--model_dir", type=str, help="model directory for generation of imaginary samples", default="out")
    parser.add_argument("--directory", type=str, help="run directory", default="out")


    args = parser.parse_args(raw_args)


    # Create the environment
    env, _ = env_factory(args.environment, args.model_class, args.seed, args.model_dir)

    # Set random seed for reproducibility
    seed = seed_all(args.seed, env=env)
    
    # Extract state and action dimensions
    state_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0]     

    # Initialize TD3 agent 
    agent = TD3(state_dim=state_dim, action_dim=action_dim, actor_hidden_dim=[300,300], critic_hidden_dim=[200,200,200], tau=0.005, policy_noise=0.1) 
    agent.exploration_noise = ExplorationNoise(action_dim=action_dim)  

    # Create a directory to save policies if it doesn't exist
    if not os.path.exists(f"./{args.directory}/saved_policy/{args.model_class}"):
        os.makedirs(f"./{args.directory}/saved_policy/{args.model_class}")

    train(env, agent, args.timesteps, args.starting_step, Path(args.directory), seed, args.model_class)




def train(env, agent, timesteps, starting_step, directory, seed, model_class):


    freq = 10000
    state, _ = env.reset(seed=seed)
    done, terminated = False, False
    


    for timestep in trange(1,timesteps+1):

        agent.update()
        # Exploration phase
        if len(agent.replay_buffer) < starting_step:
            action = env.action_space.sample()

        else: 
            action = agent.select_action_pi(state)

        next_state, reward, terminated, done, _ = env.step([action])

        # Add experience to the replay buffer
        agent.add_to_buffer(state, action, reward, next_state, terminated, done)
        state = next_state


        if done or terminated:
            state, _ = env.reset(seed=seed)
            done, terminated = False, False


        if timestep % freq == 0 :

            # Save the policy at regular intervals
            agent.save(f"./{str(directory)}/saved_policy/{model_class}/episode{int(timestep/freq)}_seed{seed}")


    print(f"training complete for seed {seed}")




if __name__ == '__main__':
    main()


