import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(THIS_DIR)
sys.path.append(PARENT_DIR)


from train_PICEM import main as train
from time import time



timesteps = 2000000

def train_all(seed, timesteps, model_class, env,  directory):



      
    train([f'--model_class={model_class}',
    	   f'--environment={env}',
           f'--timesteps={timesteps}',            
           f'--seed={seed}',  
           f'--directory={directory}'])   



   
          
if __name__ == "__main__":
     

    for s in range(10):        

        train_all(s, 500000, 'Aphynity_pendulum', 'im_pendulum',  "imagination_pendulum")
        train_all(s,500000,  'Aphynity_pendulum', 'im_pendulumsw',  "imagination_pendulumsw")

        train_all(s,500000,  'Aphynity_ctcartpole', 'im_cartpole',  "imagination_ctcartpole")
        train_all(s, 500000,  'Aphynity_ctcartpole', 'im_cartpolesw',  "imagination_ctcartpolesw")

        train_all(s,500000,  'Aphynity_ctacrobot', 'im_ctacrobotsw',  "imagination_ctacrobotsw")
        train_all(s,2000000,  'Aphynity_ctacrobot', 'im_ctacrobot',  "imagination_ctacrobot")




