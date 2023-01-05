import isaacgym

from high_level_policy import reward_scales, world_cfg, task_inplay
assert isaacgym
import torch
from mini_gym.envs import *
import wandb
import argparse

if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from high_level_policy.envs.highlevelcontrol import HighLevelControlWrapper
    from high_level_policy import HLP_ROOT_DIR, FULL_INFO
    from high_level_policy.ppo import Runner


    parser = argparse.ArgumentParser() 
     


    # add a positional argument for the number of environments
    # the type of this argument will be an integer
    # the default value will be 2500
    parser.add_argument('--num_envs', type=int, default=2500,
                        help='the number of environments to run')

    # add an optional argument for running in headless mode
    # this argument will be a boolean value (True or False)
    # the default value will be True
    parser.add_argument('--head', action='store_true', default=False,
                        help='run in headless mode')

    # parse the command line arguments
    args = parser.parse_args()

    print(args.num_envs)
    print(args.head)

    full_info = FULL_INFO

    num_envs = args.num_envs
    headless = not args.head

    env = HighLevelControlWrapper(num_envs=num_envs, headless=headless, test=False, full_info=full_info, train_ratio=0.95, hold_out=True)

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'{task_inplay}/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{HLP_ROOT_DIR}/high_level_policy/runs").resolve(), )

    

    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/command_area/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                """, filename=".charts.yml", dedent=True)
    
    reward_scales_dict = {k: v for k, v in vars(reward_scales).items() if not (k.startswith('__'))}
    world_cfg_dict = {k: v for k, v in vars(world_cfg).items() if not (k.startswith('__'))}

    logger.log_params(rewards=reward_scales_dict, world_cfg=world_cfg_dict, path='rewards.pkl')
                
    gpu_id = 0
    runner = Runner(env, device=f"cuda:{gpu_id}")
    runner.learn(num_learning_iterations=5000, eval_freq=100, eval_expert=True)
