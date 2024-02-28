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


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, help="the environment", default="acrobot")
    parser.add_argument("--agent", type=str, help="the agent used for planning (class name)", default="PICEM")
    parser.add_argument("--agent_name", type=str, help="the agent used for planning (class name)", default="rien")    
    parser.add_argument("--seed", type=int, help="seed for the randomness", default=None)
    # Model arguments
    parser.add_argument("--model_class", type=str, help="the model class")
    parser.add_argument("--device", type=str, help="the device to load the model on", default="cuda")

    # CEM planner arguments
    parser.add_argument("--horizon", type=int, help="planning horizon", default=20)
    parser.add_argument("--action_repeat", type=int, help="action repeat", default=1)
    parser.add_argument("--receding_horizon", type=int, help="receding horizon for MPC", default=25)
    parser.add_argument("--population", type=int, help="population size for CEM", default=50)
    parser.add_argument("--pi_population", type=int, help="population size from TD3", default=50)    
    parser.add_argument("--selection", type=int, help="selected population size for CEM", default=10)
    parser.add_argument("--iterations", type=int, help="number of iterations in CEM", default=3)
    parser.add_argument("--use_oracle", type=str2bool, help="use of oracle dynamics in CEM", default=False)

    # Random policy arguments
    parser.add_argument("--sigma", type=float, help="integrated noise std", default=0.1)
    parser.add_argument("--noise_tau", type=float, help="noise response time, in number of steps", default=20)
    # Evaluation arguments
    parser.add_argument("--episodes", type=int, help="number of episodes", default=10)
    parser.add_argument("--plot_only", type=str2bool, help="plot without generating", nargs='?', const=True, default=False,)
    parser.add_argument("--directory", type=str, help="run directory", default="out")

    parser.add_argument("--PIHP", type=str2bool, help="PIHP or td3", nargs='?', const=True, default=True,)
    parser.add_argument("--td", type=str2bool, help="use Q value or not", nargs='?', const=True, default=True,)
    parser.add_argument("--alpha", type=float, help="regularization", default=0.2)

    args = parser.parse_args(raw_args)

    if not args.plot_only:

        env_test, observer_test = env_factory(args.environment)

        seed = seed_all(args.seed, env=env_test)
        #seed = args.seed
        agent = agent_factory(env_test, observer_test, args.agent, **vars(args))

        alpha = args.alpha




        ep = 10


        if args.environment == 'pendulum':
            agent.mbrl_path = f"training_model_pendulum/episode_{ep}/models/{agent.model_class}_model_{seed}.tar" 
            rl_path = f"./imagination_pendulum/saved_policy/PerfectPendulumModel/"

        elif args.environment == 'pendulumsw':
            rl_path = f"./imagination_pendulumsw/saved_policy/PerfectPendulumModel/"
            agent.mbrl_path = f"training_model_pendulumsw/episode_10/models/{agent.model_class}_model_{seed}.tar"             

        elif args.environment == 'ctcartpole':
            agent.mbrl_path = f"training_model_cartpole/episode_{3*ep}/models/{agent.model_class}_model_{seed}.tar" 
            rl_path = f"./imagination_ctcartpole/saved_policy/PerfectCartPole/"

        elif args.environment == 'ctcartpolesw':
            agent.mbrl_path = f"training_model_cartpolesw/episode_{ep}/models/{agent.model_class}_model_{seed}.tar" 
            rl_path = f"./imagination_ctcartpolesw/saved_policy/PerfectCartPole/"

        elif args.environment == 'ctacrobot':
            agent.mbrl_path = f"training_model_acrobot/episode_{ep}/models/{agent.model_class}_model_{seed}.tar" 
            rl_path = f"./imagination_ctacrobot/saved_policy/Perfectacrobot/"

        elif args.environment == 'ctacrobotsw':
            agent.mbrl_path = f"training_model_acrobotsw/episode_30/models/{agent.model_class}_model_{seed}.tar" 
            rl_path = f"./imagination_ctacrobotsw/saved_policy/Perfectacrobot/"

        if not ("Perfect" in agent.model_class) :
            agent.load_dynamics(agent.model_class,agent.mbrl_path)
        agent.pi_population = args.pi_population  #population
        agent.Q = args.td
        agent.alpha = alpha

        if args.PIHP:
            PIHP_reward = test_PIHP( agent, int(args.episodes), rl_path, seed, env_test)
            save_data_PIHP(PIHP_reward , args.environment, args.agent_name, agent, seed,  Path(args.directory), args.population, args.pi_population, args.selection, agent.Q, args.receding_horizon, args.horizon )
            plot_all(directory=Path(args.directory), filenames="PIHP_data.csv" )
        else:
            TD3_reward = test_TD3(agent, int(args.episodes), rl_path, seed, env_test )
            save_data_TD3( TD3_reward, args.environment, args.agent_name, agent, seed,  Path(args.directory))
            plot_all(directory=Path(args.directory), filenames="TD3_data.csv" )  

    else:
        if args.PIHP:
            plot_all(directory=Path(args.directory), filenames="PIHP_data.csv" )
        else:
            plot_all(directory=Path(args.directory), filenames="TD3_data.csv" )
    
    #plot_all(directory=Path(args.directory), filenames="training_data.csv" )
    #if raw_args is None:
    plt.close()

def test_PIHP(agent, episodes, rl_path, seed, env_test):


    PIHP_reward = []



        
    agent.rl.load(rl_path + f"episode{episodes}_seed{seed}")
    episode_rewards = []
    for i in range(10):

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
        episode_rewards.append(ep_reward)
    
    PIHP_reward.append(np.nanmean(episode_rewards))

    return PIHP_reward

def test_TD3( agent, episodes, rl_path, seed, env_test):

    TD3_reward = []


    for episode in trange(1,episodes+1):
        
        agent.rl.load(rl_path + f"episode{episode}_seed{seed}")
        episode_rewards = []

        for i in range(1):

            #Testing with TD3
        
            agent.test = True

            agent.env = env_test
            state, _ = agent.env.reset()
            agent.reset()
            done, terminated = False, False
            ep_reward = 0

            while not (done or terminated):

                action = agent.actfree(state)
                next_state, reward, terminated, done, info = env_test.step([action])
                state = next_state
                ep_reward += reward

            episode_rewards.append(ep_reward)

        
        TD3_reward.append(np.nanmean(episode_rewards))


    return TD3_reward




def plot_last_episode(observer, agent):
    # Plot results
    df = pd.DataFrame(observer.episodes[-1])
    try:
        agent.plot(df)
    except AttributeError:
        pass



def save_data_PIHP(PIHP_reward, env, agent_name, agent, seed, directory, population, pi_population, selection, wq, rh, h, filename="PIHP_data.csv"):

    PIHP_results = [{
        "env": env,
        "agent": agent_name,
        "episode": episode+1,
        "seed": seed,
        "total_reward": rewards,
    } for episode, rewards in enumerate(PIHP_reward)]
    PIHP_results = pd.DataFrame.from_records(PIHP_results)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, 'a') as f:
        PIHP_results.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)



def save_data_TD3(TD3_reward,env, agent_name, agent, seed, directory, filename="TD3_data.csv"):



    TD3_results = [{
        "env": env,
        "agent": agent_name,
        "episode": episode+1,
        "seed": seed,
        "total_reward": rewards,
    } for episode, rewards in enumerate(TD3_reward)]
    TD3_results = pd.DataFrame.from_records(TD3_results)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, 'a') as f:
        TD3_results.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)




def plot_all(directory, filenames="data.csv"):
    
    df = pd.read_csv(str(directory / filenames))
    #df = df[df.episode != 0]
    df["episode"] *= 10 #*(10-9*i)
    #df["episode"] -= 500


    for env in ["pendulum", "pendulumsw", "ctcartpole", "ctcartpolesw", "ctacrobotsw"]:
        df_ = df[df.env == env]
        df_ = df_[~df_.agent.isin(['agent'])].apply(pd.to_numeric, errors='ignore')    
        df_ = df_.sort_values(by="agent")

        try:
            for field in ["total_reward"]:
                fig, ax = plt.subplots()
                sns.lineplot(x="episode", y=field, ax=ax, hue="agent", data=df_)
                plt.xlabel('x1000 steps')
                plt.legend(loc="lower right")
                field_path = directory / "{}.pdf".format(env+filenames)
                fig.savefig(field_path, bbox_inches='tight')

                field_path = directory / "{}.png".format(filenames)
                fig.savefig(field_path, bbox_inches='tight')

                print("Saving {} plot to {}".format(field, field_path))
        except ValueError as e:
            print(e)

    

if __name__ == '__main__':
    main()


