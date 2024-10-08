import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(THIS_DIR)
sys.path.append(PARENT_DIR)


from train_model import main as train




epochs = 4000
lr = 0.001



def train_model(seed, model_class, lambda_0, tau,  env, receding_horizon, horizon, population, selection, model_fit_frequency, timesteps, directory):     

    train([f'--model_class={model_class}',
    	   f'--environment={env}',
           f'--epochs={epochs}',
           f'--lr={lr}',
           f'--timesteps={timesteps}',           
           f'--receding_horizon={receding_horizon}',
           f'--horizon={horizon}',
           f'--population={population}',
           f'--selection={selection}', 
           f'--lambda_0={lambda_0}',
           f'--aph_tau={tau}', 
           f'--seed={seed}',  
           f'--model_fit_frequency={model_fit_frequency}',           
           f'--directory={directory}'])   
    

          
if __name__ == "__main__":
     

    for s in range(10):        



        ######################################     Pendulum   #######################################################


        train_model(s, 'PhIHP_Pendulum', 1000.0, 1000.0, 'pendulum', 15, 30, 500, 20, 200, 2000,   "training_model_pendulum") 
 
                                    ####### Swingup #######
        
        train_model(s, 'PhIHP_Pendulum', 1000.0, 1000.0, 'pendulumsw', 15, 30, 500, 20, 500, 5000,   "training_model_pendulumsw") 


        ######################################     CARTPOLE   #######################################################


        train_model(s, 'PhIHP_Cartpole', 1000.0, 100000.0, 'ctcartpole', 10, 50, 100, 10, 500, 5000,  "training_model_cartpole")

                                    ####### Swingup #######
        
        train_model(s, 'PhIHP_Cartpole',1000.0, 100000.0, 'ctcartpolesw', 10, 50, 500, 20, 500, 5000,  "training_model_cartpolesw")  

          
        ######################################     Acrobot   #######################################################


        train_model(s, 'PhIHP_Acrobot', 100.0, 100000.0, 'ctacrobot', 10, 30, 700, 20,  500, 5000,  "training_model_acrobot")

                                    ####### Swingup ####### 
        
        train_model(s, 'PhIHP_Acrobot_sw', 1000.0, 100000.0, 'ctacrobotsw', 10, 30, 500, 20, 500, 15000,  "training_model_acrobotsw")     
