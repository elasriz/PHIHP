import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(THIS_DIR)
sys.path.append(PARENT_DIR)


from train_policy import main as train

def train_all(seed, timesteps, model_class, env,  directory, model_dir):



      
    train([f'--model_class={model_class}',
    	   f'--environment={env}',
           f'--timesteps={timesteps}',            
           f'--seed={seed}',  
           f'--model_dir={model_dir}',  
           f'--directory={directory}'])   



   
          
if __name__ == "__main__":
     

    for s in range(10):        

        train_all(s, 500000, 'PhIHP_Pendulum', 'im_pendulum',  "imagination_pendulum", "training_model_pendulum/episode_10" )
        train_all(s, 500000,  'PhIHP_Pendulum', 'im_pendulumsw',  "imagination_pendulumsw", "training_model_pendulumsw/episode_10")

        train_all(s, 500000,  'PhIHP_Cartpole', 'im_cartpole',  "imagination_ctcartpole", "training_model_cartpole/episode_10")
        train_all(s, 500000,  'PhIHP_Cartpole', 'im_cartpolesw',  "imagination_ctcartpolesw", "training_model_cartpolesw/episode_10")

        train_all(s, 500000,  'PhIHP_Acrobot', 'im_ctacrobot',  "imagination_ctacrobot", "training_model_acrobot/episode_10")
        train_all(s, 2000000, 'PhIHP_Acrobot_sw', 'im_ctacrobotsw',  "imagination_ctacrobotsw", "training_model_acrobotsw/episode_30")
        




