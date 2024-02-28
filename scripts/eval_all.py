import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(THIS_DIR)
sys.path.append(PARENT_DIR)


from EVAL_PIHP import main as test
from time import time





def test_model(model_class,seed, alpha, episodes, env,  receding_horizon, horizon, population, pi_population,selection, PIHP, td, agent_name, directory):

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
           f'--PIHP={PIHP}',        
           f'--td={td}',                  
           f'--directory={directory}'])     

         
if __name__ == "__main__":

 

    for seed in range(10):              

            test_model("Aphynity_pendulum", seed, 1.5, 50, 'pendulumsw', 1, 4, 200, 20, 15, True, True, "PhIHP", "results/pendulumsw")
            test_model("Aphynity_pendulum", seed, 1.5, 50, 'pendulum', 1, 4, 200, 20, 15, True, True, "PhIHP", "results/pendulum")
            test_model("Aphynity_ctcartpole", seed, 0.2, 50, 'ctcartpole', 1, 3, 200, 20, 15, True, True, "PhIHP", "results/cartpolesw")
            test_model("Aphynity_ctcartpole", seed , 0.03, 50, 'ctcartpolesw', 1, 4, 200, 20, 15, True, True, "PhIHP", "results/cartpole") 

            test_model("Aphynity_ctacrobot", seed, 0.8, 200, 'ctacrobotsw', 1, 3, 200, 20, 15, True, True,"PhIHP", "results/acrobotsw")
            test_model("Aphynity_ctacrobot", seed, 0.8, 50, 'ctacrobot', 1, 3, 200, 20, 15, True, True,"PhIHP", "results/acrobot")
                       


