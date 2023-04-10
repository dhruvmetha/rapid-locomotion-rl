from isaacgym import gymapi
from PIL import Image
from ml_logger import logger
from high_level_policy import *


import numpy as np
from matplotlib import patches as pch
import argparse

def create_env(num_envs, headless, full_info, train_ratio=0.0):
    from high_level_policy.envs.highlevelcontrol import HighLevelControlWrapper
    env = HighLevelControlWrapper(num_envs=num_envs, headless=headless, test=True, full_info=full_info, train_ratio=train_ratio)
    return env

if __name__ == "__main__":
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_envs', type=int, default=25,
                        help='the number of environments to run')
    
    parser.add_argument('--iter', type=int, default=10000,
                        help='the number of environments to run')
    
    parser.add_argument('--head', action='store_true', default=False,
                        help='run in headless mode')

    args = parser.parse_args()

    num_envs = args.num_envs
    iters = args.iter
    headless = not args.head
    
    env = create_env(num_envs, headless, full_info=False)
    num_eval_steps = iters
    obs = env.reset()





    # from world_model.scripts.runner import Runner

    # Runner(env).run(num_iterations=iters)

    import torch
    actions = torch.zeros(num_envs, 3)
    actions[:, :3] = 2. * torch.rand(num_envs, 3) - 1. 
    for i in tqdm(range(num_eval_steps)):
        obs, rew, done, info = env.step(actions)

        if i%10 == 0:
            actions[:, :3] = 2. * torch.rand(num_envs, 3) - 1.

# Steps
## 1. random actions for initial data collection
##### 1a. train a transformer model for the privileged information prediction (encoder, mse loss with truth)
## 2. 
        

