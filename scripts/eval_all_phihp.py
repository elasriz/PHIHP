import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(THIS_DIR)
sys.path.append(PARENT_DIR)


from eval_PHIHP import main as test



def test_agent(model_class,seed, alpha, episodes, env,  receding_horizon, horizon, population, pi_population,selection, PHIHP_role, use_Q, agent_name, model_path, rl_path, directory):

    test([f'--model_class={model_class}',
    	   f'--environment={env}',
    	   f'--agent_name={agent_name}',            
           f'--episodes={episodes}',
           f'--receding_horizon={receding_horizon}',
           f'--horizon={horizon}',
           f'--population={population}',
           f'--pi_population={pi_population}', 
           f'--selection={selection}',                     
           f'--seed={seed}',  
           f'--alpha={alpha}', 
           f'--PHIHP_role={PHIHP_role}',        
           f'--use_Q={use_Q}', 
           f'--model_path={model_path}',
           f'--rl_path={rl_path}',
           f'--directory={directory}'])     

         
if __name__ == "__main__":

 

    for seed in range(10):              


        ######################################     Pendulum   #######################################################
            
            test_agent("PhIHP_Pendulum", seed, 1.5, 50, 'pendulum', 1, 5, 200, 20, 10, True, True, "PhIHP", 
                       f"training_model_pendulum/episode_10/models/PhIHP_Pendulum_model_{seed}.tar" ,
                        f"./imagination_pendulum/saved_policy/PhIHP_Pendulum/",
                       "results/pendulum")

                                    ####### Swingup #######
                      
            test_agent("PhIHP_Pendulum", seed, 1.5, 50, 'pendulumsw', 1, 5, 200, 20, 10, True, True, "PhIHP", 
                       f"training_model_pendulumsw/episode_10/models/PhIHP_Pendulum_model_{seed}.tar" ,
                        f"./imagination_pendulumsw/saved_policy/PhIHP_Pendulum/",                  
                       "results/pendulumsw")

        ######################################     CARTPOLE   #######################################################

            test_agent("PhIHP_Cartpole", seed, 0.2, 50, 'ctcartpole', 1, 4, 200, 20, 10, True, True, "PhIHP", 
                       f"training_model_cartpole/episode_10/models/PhIHP_Cartpole_model_{seed}.tar" ,
                        f"./imagination_ctcartpole/saved_policy/PhIHP_Cartpole/",                        
                       "results/cartpole")      

                                    ####### Swingup #######
            
            test_agent("PhIHP_Cartpole", seed , 0.03, 50, 'ctcartpolesw', 1, 6, 200, 20, 10, True, True, "PhIHP", 
                       f"training_model_cartpolesw/episode_10/models/PhIHP_Cartpole_model_{seed}.tar" ,
                        f"./imagination_ctcartpolesw/saved_policy/PhIHP_Cartpole/",                          
                       "results/cartpolesw")

        ######################################     Acrobot   #######################################################

            test_agent("PhIHP_Acrobot", seed, 0.8, 50, 'ctacrobot', 1, 4, 200, 20, 10, True, True,"PhIHP",
                       f"training_model_acrobot/episode_10/models/PhIHP_Acrobot_model_{seed}.tar" ,
                        f"./imagination_ctacrobot/saved_policy/PhIHP_Acrobot/", 
                        "results/acrobot")

                                    ####### Swingup ####### 
        
            test_agent("PhIHP_Acrobot_sw", seed, 0.8, 200, 'ctacrobotsw', 1, 3, 200, 20, 10, True, True,"PhIHP", 
                       f"training_model_acrobotsw/episode_30/models/PhIHP_Acrobot_sw_model_{seed}.tar" ,
                        f"./imagination_ctacrobotsw/saved_policy/PhIHP_Acrobot_sw/",                        
                       "results/acrobotsw")
            
                       


