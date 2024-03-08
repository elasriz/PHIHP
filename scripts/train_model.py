import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(THIS_DIR)
sys.path.append(PARENT_DIR)


from pathlib import Path
from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from source.utils import str2bool, seed_all

sns.set_style("whitegrid")

from source.agents import agent_factory
from source.envs import env_factory

import argparse
import time

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
    # Loss arguments
    parser.add_argument("--lambda_0", type=float, help="", default=1000.0)
    parser.add_argument("--Niter", type=int, help="", default=3)
    parser.add_argument("--aph_tau", type=float, help="", default=100000.0)
    # CEM planner arguments
    parser.add_argument("--horizon", type=int, help="planning horizon for MPC", default=20)
    parser.add_argument("--action_repeat", type=int, help="action repeat", default=1)
    parser.add_argument("--receding_horizon", type=int, help="receding horizon for MPC", default=25)
    parser.add_argument("--population", type=int, help="population size for CEM", default=300)
    parser.add_argument("--selection", type=int, help="selected population size for CEM", default=20)
    parser.add_argument("--iterations", type=int, help="number of iterations in CEM", default=3)
    parser.add_argument("--model_fit_frequency", type=int, help="frequency of model fitting", default=200)
    parser.add_argument("--use_oracle", type=str2bool, help="use of oracle dynamics in CEM", default=False)

    # Exploration arguments
    parser.add_argument("--sigma", type=float, help="integrated noise std", default=0.1)
    parser.add_argument("--noise_tau", type=float, help="noise response time, in number of steps", default=20)
    # other arguments
    parser.add_argument("--timesteps", type=int, help="number of timesteps for training", default=1000)    
    parser.add_argument("--plot_only", type=str2bool, help="plot without generating", nargs='?', const=True, default=False,)
    parser.add_argument("--directory", type=str, help="run directory", default="out")

    args = parser.parse_args(raw_args)

    if not args.plot_only:

        
        # Create the environment and observer 
        env, observer = env_factory(args.environment)
        env_test, observer_test = env_factory(args.environment)
        # Create the agent
        agent = agent_factory(env, observer, args.agent, **vars(args))

        # Set the random seed for reproducibility
        seed = seed_all(args.seed, env=env)

        # Train the agent
        print(f"Training agent with model {args.model_class}")
        training_rewards = train(env, observer, env_test, observer_test, agent, args.timesteps, Path(args.directory), seed, args.model_fit_frequency)

        save_data(training_rewards, agent, observer, seed,  Path(args.directory))

    plot_all(args.model_fit_frequency, directory=Path(args.directory), filenames="training_data.csv" )
    plt.close()


def evaluate(env_test, observer_test, agent, seed, ev_freq = 10):
    """ Evaluate the performance of a trained agent """

    
    
    # Set the agent
    agent.test = True
    agent.env = env_test
    observer_test.reset()
    agent.reset()
    
    
    episode_rewards = []

     
    for _ in range(ev_freq):
        
        # Reset the test environment
        state, _ = env_test.reset()
        done, terminated = False, False
        ep_reward = 0

        # Run the evaluation episode until termination
        while not (done or terminated):

            action = agent.act(state)
            observer_test.log(env_test, state, action)
            next_state, reward, terminated, done, _ = env_test.step([action])
            state = next_state
            ep_reward += reward
        
        episode_rewards.append(ep_reward)

    
    # Plot the last episode 
    plot_last_episode(observer_test, agent)

    return np.nanmean(episode_rewards)



def train(env, observer, env_test, observer_test, agent, timesteps, directory, seed,  mod_fit_freq):
    """ Train a model-based agent """

    #Initialization
    training_rewards = []
    episode = 1

    
    
    # Set the agent
    agent.test = False
    agent.env = env
    observer.reset()
    agent.reset()
    
    # Set a directory for each episode
    agent.directory = directory / f"episode_{episode}/"
    
    state, _ = env.reset(seed=seed)
    done, terminated = False, False



    for timestep in trange(1,timesteps+1):

        action = agent.act(state)
        observer.log(env, state, action)
        next_state, reward, terminated, done, _ = env.step([action])
        state = next_state

        if done or terminated:
            #reset the environment
            state, _ = env.reset(seed=seed)
            done, terminated = False, False
            agent.reset()
            observer.reset()

        # Evaluate the agent's performance
        if timestep % mod_fit_freq == 0 :
            ep_reward = evaluate(env_test, observer_test, agent, seed)
            training_rewards.append(ep_reward)

            state, _ = env.reset(seed=seed)
            agent.test = False
            agent.env = env
            observer.reset()
            agent.reset()
            done, terminated = False, False

            

            # Update the directory 
            episode += 1
            agent.directory = directory / f"episode_{episode}/"

        
        plt.close()

    return  training_rewards



def plot_last_episode(observer, agent):
    # Plot results
    df = pd.DataFrame(observer.episodes[-1])
    try:
        agent.plot(df)
    except AttributeError:
        pass
    plt.close()


def save_data(training_rewards, agent, observer, seed, directory, filename="training_data.csv"):
    # Save data
    observer.append_to(directory / "data/dataset_{}.csv".format(agent.dynamics_model.__class__.__name__))
    # Save results
    agent_name = agent.dynamics_model.__class__.__name__ 
    training_results = [{
        "agent": agent_name,
        "episode": episode,
        "seed": seed,
        "total_reward": training_rewards,
    } for episode, training_rewards in enumerate(training_rewards)]
    training_results = pd.DataFrame.from_records(training_results)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, 'a') as f:
        training_results.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)


def plot_all(freq, directory, filenames="training_data.csv"):
    
    df = pd.read_csv(str(directory / filenames))

    df["episode"] *= freq / 1000

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


