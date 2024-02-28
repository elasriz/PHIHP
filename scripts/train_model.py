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

import argparse
import time
import copy

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, help="the environment", default="acrobot")
    parser.add_argument("--agent", type=str, help="the agent used for planning (class name)", default="CEMPlanner")
    parser.add_argument("--seed", type=int, help="seed for the randomness", default=None)
    # Model arguments
    parser.add_argument("--model_class", type=str, help="the model class")
    parser.add_argument("--device", type=str, help="the device to load the model on", default="cuda")
    # Training
    parser.add_argument("--epochs", type=int, help="number of epochs", default=4000)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
    # APHYNITY parameters
    parser.add_argument("--lambda_0", type=float, help="", default=1000.0)
    parser.add_argument("--Niter", type=int, help="", default=3)
    parser.add_argument("--aph_tau", type=float, help="", default=1000000.0)
    # CEM planner arguments
    parser.add_argument("--horizon", type=int, help="planning horizon", default=20)
    parser.add_argument("--action_repeat", type=int, help="action repeat", default=1)
    parser.add_argument("--receding_horizon", type=int, help="receding horizon for MPC", default=25)
    parser.add_argument("--population", type=int, help="population size for CEM", default=300)
    parser.add_argument("--selection", type=int, help="selected population size for CEM", default=20)
    parser.add_argument("--iterations", type=int, help="number of iterations in CEM", default=3)
    parser.add_argument("--model_fit_frequency", type=int, help="frequency of model fitting", default=200)
    parser.add_argument("--use_oracle", type=str2bool, help="use of oracle dynamics in CEM", default=False)

    # Random policy arguments
    parser.add_argument("--sigma", type=float, help="integrated noise std", default=0.1)
    parser.add_argument("--noise_tau", type=float, help="noise response time, in number of steps", default=20)
    # Evaluation arguments
    parser.add_argument("--episodes", type=int, help="number of episodes", default=10)
    parser.add_argument("--timesteps", type=int, help="number of timesteps", default=1000)    
    parser.add_argument("--plot_only", type=str2bool, help="plot without generating", nargs='?', const=True, default=False,)
    parser.add_argument("--directory", type=str, help="run directory", default="out")

    args = parser.parse_args(raw_args)

    if not args.plot_only:
        env, observer = env_factory(args.environment)
        seed = seed_all(args.seed, env=env)

        agent = agent_factory(env, observer, args.agent, **vars(args))


        print(f"Training agent with model {args.model_class}")
        training_time = train(env, observer, agent, args.timesteps, Path(args.directory), seed, args.model_fit_frequency)
        save_data(training_time, agent, observer, seed,  Path(args.directory))
    plot_all(directory=Path(args.directory), filenames="training_data.csv" )
    #if raw_args is None:
    plt.close()

def train(env, observer, agent, timesteps, directory, seed,  mod_fit_freq):



    state, _ = env.reset(seed=seed)
    agent.test = False
    agent.env = env
    observer.reset()
    agent.reset()
    
    done, terminated = False, False


    start_time = time.process_time()
    training_time = []
    episode = 1

    for timestep in trange(1,timesteps+1):


        if timestep % mod_fit_freq == 0 :


            training_time.append(time.process_time() - start_time)
            
            agent.directory = directory / f"episode_{episode}/"
            
            #plot_last_episode(observer, agent)
            plt.close()
            episode +=1

        action = agent.act(state)
        observer.log(env, state, action)
        next_state, reward, terminated, done, _ = env.step([action])
        state = next_state


        if done or terminated:
            state, _ = env.reset(seed=seed)
            done, terminated = False, False
            agent.reset()
            observer.reset()

    return  training_time



def plot_last_episode(observer, agent):
    # Plot results
    df = pd.DataFrame(observer.episodes[-1])
    #print(df)
    try:
        agent.plot(df)
    except AttributeError:
        pass
    plt.close()


def save_data(training_time, agent, observer, seed, directory, filename="training_data.csv"):
    # Save data
    observer.append_to(directory / "data/dataset_{}.csv".format(agent.dynamics_model.__class__.__name__))
    # Save results
    agent_name = agent.dynamics_model.__class__.__name__ +"_" +str(agent.population) +"_"+ str(agent.horizon) +"_"+ str(agent.receding_horizon)
    training_results = [{
        "agent": agent_name,
        "episode": episode,
        "seed": seed,
        "Trainig_Time": training_time,
    } for episode, training_time in enumerate(training_time)]
    training_results = pd.DataFrame.from_records(training_results)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, 'a') as f:
        training_results.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)


def plot_all(directory, filenames="data.csv"):
    
    df = pd.read_csv(str(directory / filenames))

    df["episode"] *= 0.2 #*(10-9*i)

    df = df[~df.agent.isin(['agent'])].apply(pd.to_numeric, errors='ignore')    
    df = df.sort_values(by="agent")

    try:
        for field in ["total_reward"]:
            fig, ax = plt.subplots()
            sns.lineplot(x="episode", y=field, ax=ax, hue="agent", data=df)
            plt.xlabel('x1000 steps')
            plt.legend(loc="lower right")
            field_path = directory / "{}.pdf".format(filenames)
            fig.savefig(field_path, bbox_inches='tight')
            field_path = directory / "{}.png".format(filenames)
            fig.savefig(field_path, bbox_inches='tight')
            print("Saving {} plot to {}".format(field, field_path))
    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main()


