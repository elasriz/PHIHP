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

sns.set_style("whitegrid")

from source.agents import agent_factory
from source.envs import env_factory, plot_trajectories
from source.TD3 import TD3, ExplorationNoise
import argparse
import time


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, help="the environment", default="acrobot")
    parser.add_argument("--agent", type=str, help="the agent used for planning (class name)", default="PICEM_train")
    parser.add_argument("--seed", type=int, help="seed for the randomness", default=None)
    # Model arguments
    parser.add_argument("--model_class", type=str, help="the model class")
    parser.add_argument("--device", type=str, help="the device to load the model on", default="cuda")

    # Evaluation arguments
    parser.add_argument("--timesteps", type=int, help="number of timesteps", default=1000)    
    parser.add_argument("--plot_only", type=str2bool, help="plot without generating", nargs='?', const=True, default=False,)
    parser.add_argument("--directory", type=str, help="run directory", default="out")

    args = parser.parse_args(raw_args)


    if not args.plot_only:

        env, observer = env_factory(args.environment, args.model_class, args.seed)
        seed = seed_all(args.seed, env=env)
        

        agent = agent_factory(env, observer, args.agent, **vars(args))


        if not os.path.exists(f"./{args.directory}/saved_policy/{agent.model_class}"):
            os.makedirs(f"./{args.directory}/saved_policy/{agent.model_class}")

    

        training_time = train(env, observer, agent, args.timesteps, Path(args.directory), seed)
        save_data(training_time, agent, observer, seed,  Path(args.directory))

    #plot_all(directory=Path(args.directory), filenames="training_data.csv" )
    #if raw_args is None:
    plt.close()

def train(env, observer, agent, timesteps, directory, seed):


    start_time = time.process_time()
    training_times = [] 

    freq = 10000


    state, _ = env.reset(seed=seed)

    done, terminated = False, False
    agent.reset()
    agent.test = False
    agent.env = env


    for timestep in trange(1,timesteps+1):
    
        action = agent.act(state)
        next_state, reward, terminated, done, _ = env.step([action])


        agent.rl.add_to_buffer(state, action, reward, next_state, terminated, done)
        state = next_state


        if done or terminated:
            state, _ = env.reset(seed=seed)
            done, terminated = False, False
            agent.reset()

        if timestep % freq == 0 :

            agent.rl.save(f"./{str(directory)}/saved_policy/{agent.model_class}/episode{int(timestep/freq)}_seed{seed}")
            training_times.append(time.process_time() - start_time)
            
        plt.close()

    return training_times



def save_data(training_times, agent, observer, seed, directory, filename="training_data.csv" ):

    training_results = [{
        "agent": agent.dynamics_model.__class__.__name__,
        "episode": episode,
        "seed": seed,
        "training_times": training_time,
    } for episode, training_time in enumerate(training_times)]
    training_results = pd.DataFrame.from_records(training_results)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, 'a') as f:
        training_results.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)


if __name__ == '__main__':
    main()


