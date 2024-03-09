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




from source.utils import str2bool
from source.agents import agent_factory
from source.envs import env_factory

sns.set_style("whitegrid")
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


    # Create the test environment and observer
    env_test, observer_test = env_factory(args.environment)

    # set the seed to point the saved policy 
    seed = args.seed

    # Create the agent
    agent = agent_factory(env_test, observer_test, args.agent, **vars(args))


    # set the path to the saved policy
    rl_path = args.rl_path

    # Load the trained dynamics model
    agent.load_dynamics(agent.model_class,args.model_path)


    # Test and save results
    PHIHP_reward = test_PHIHP( agent, env_test, int(args.episodes), rl_path, seed, Path(args.directory))
    save_data_PHIHP(PHIHP_reward , args.environment, args.agent_name, seed,  Path(args.directory) )
    plot_all(args.environment, directory=Path(args.directory), filenames="PHIHP_data.csv" )


    plt.close()

def test_PHIHP(agent, env_test, episodes, rl_path, seed, dir):


    PHIHP_reward = []

    for episode in trange(1,episodes+1):
   
        agent.rl.load(rl_path + f"episode{episode}_seed{seed}")
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

        PHIHP_reward.append(np.nanmean(episode_rewards))

    return PHIHP_reward



def save_data_PHIHP(PHIHP_reward, env, agent_name, seed, directory, filename="PHIHP_data.csv"):

    PHIHP_results = [{
        "env": env,
        "agent": agent_name,
        "episode": episode+1,
        "seed": seed,
        "total_reward": rewards,
    } for episode, rewards in enumerate(PHIHP_reward)]
    PHIHP_results = pd.DataFrame.from_records(PHIHP_results)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, 'a') as f:
        PHIHP_results.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)





def plot_all(env, directory, filenames="data.csv"):
    
    df = pd.read_csv(str(directory / filenames))

    df["episode"] *= 10

    df = df[~df.agent.isin(['agent'])].apply(pd.to_numeric, errors='ignore')    
    df = df.sort_values(by="agent")
    try:
        for field in ["total_reward"]:
            fig, ax = plt.subplots()
            sns.lineplot(x="episode", y=field, ax=ax, hue="agent", data=df)
            plt.xlabel('x1000 steps')
            plt.legend(loc="lower right")
            field_path = directory / "{}_{}.pdf".format(env,filenames[:-4])
            fig.savefig(field_path, bbox_inches='tight')
            field_path = directory / "{}_{}.png".format(env,filenames[:-4])
            fig.savefig(field_path, bbox_inches='tight')
            print("Saving {} plot to {}".format(field, field_path))
    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main()


