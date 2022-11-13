import isaacgym

from high_level_policy import reward_scales, world_cfg
assert isaacgym
import torch
from mini_gym.envs import *
import wandb
import argparse

if __name__ == '__main__':
    from pathlib import Path
    from ml_logger import logger
    from high_level_policy.envs.highlevelcontrol import HighLevelControlWrapper
    from high_level_policy import HLP_ROOT_DIR
    from high_level_policy.ppo import Runner


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--iter', type=int, default=5000, help='an integer for the accumulator')
    # parser.add_argument('--envs', 
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    # args = parser.parse_args()
    # print(args.accumulate(args.integers))


    num_envs = 4096
    env = HighLevelControlWrapper(num_envs=num_envs, headless=True, test=False)

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'rapid-locomotion/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{HLP_ROOT_DIR}/high_level_policy/runs").resolve(), )

    wandb.init('legged_navigation')

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
