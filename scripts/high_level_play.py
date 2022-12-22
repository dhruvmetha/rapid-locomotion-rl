from ml_logger import logger
from high_level_policy import *

import numpy as np
# from matplotlib import pyplot as plt
from matplotlib import patches as pch
# import matplotlib.animation as animation
import pickle
import argparse

def load_env(headless=False):
    from ml_logger import logger
    print(logger.glob("*"))
    print(logger.prefix)

    params = logger.load_pkl('parameters.pkl')

    if 'kwargs' in params[0]:
        deps = params[0]['kwargs']

        from high_level_policy.ppo.ppo import PPO_Args
        from high_level_policy.ppo.actor_critic import AC_Args
        from high_level_policy.ppo import RunnerArgs

        AC_Args._update(deps)
        PPO_Args._update(deps)
        RunnerArgs._update(deps)
    
    # Cfg.env.num_recording_envs = 1

    # load policy
    from ml_logger import logger
    from high_level_policy.ppo.actor_critic import ActorCritic

    print(env.num_obs_history)

    actor_critic = ActorCritic(
        num_obs=env.num_obs, 
        num_privileged_obs=env.num_privileged_obs,
        num_obs_history=(env.num_obs+12) * \
                        ROLLOUT_HISTORY,
        num_actions=env.num_actions)

    # print(actor_critic)

    # print(logger.prefix)
    # print(logger.glob("*"))
    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    policy_student = actor_critic.act_inference
    policy_expert = actor_critic.act_inference_expert

    return policy_expert, policy_student

def get_patch_set(obs_truth, priv_obs_truth, priv_obs_pred):
    pos_rob_eval, rot_rob_eval = obs_truth[:2], obs_truth[2:6]
    angle_rob_eval = torch.rad2deg(torch.atan2(2.0*(rot_rob_eval[0]*rot_rob_eval[1] + rot_rob_eval[3]*rot_rob_eval[2]), 1. - 2.*(rot_rob_eval[1]*rot_rob_eval[1] + rot_rob_eval[2]*rot_rob_eval[2])))

    patch_set = []
    patch_set.append(pch.Rectangle(pos_rob_eval.cpu().numpy() - np.array([0.588, 0.22]), width=0.588, height=0.22, angle=angle_rob_eval.cpu(), rotation_point='center', facecolor='green', label='robot'))
    patch_set.append(pch.Rectangle(pos_rob_eval.cpu().numpy() - np.array([0.588, 0.22]), width=0.588, height=0.22, angle=angle_rob_eval.cpu(), rotation_point='center', facecolor='green', label='robot'))
    patch_set.append(pch.Rectangle(pos_rob_eval.cpu().numpy() - np.array([0.588, 0.22]), width=0.588, height=0.22, angle=angle_rob_eval.cpu(), rotation_point='center', facecolor='green', label='robot'))
    
    for i in range(3):
        j = i*8
        pos, pos_pred = priv_obs_truth[j:j+2], priv_obs_pred[j:j+2]
        rot, rot_pred = priv_obs_truth[j+2:j+6], priv_obs_pred[j+2:j+6]
        size, size_pred = priv_obs_truth[j+6:j+8], priv_obs_pred[j+6:j+8]

        angle = torch.rad2deg(torch.atan2(2.0*(rot[0]*rot[1] + rot[3]*rot[2]), 1. - 2.*(rot[1]*rot[1] + rot[2]*rot[2])))
        angle_pred = torch.rad2deg(torch.atan2(2.0*(rot_pred[0]*rot_pred[1] + rot_pred[3]*rot_pred[2]), 1. - 2.*(rot_pred[1]*rot_pred[1] + rot_pred[2]*rot_pred[2])))

        patch_set.append(pch.Rectangle(pos.cpu() - size.cpu()/2, *(size.cpu()), angle=angle.cpu(), rotation_point='center', facecolor='red', label=f'true_mov_{i}'))
        patch_set.append(pch.Rectangle(pos_pred.cpu() - size_pred.cpu()/2, *(size_pred.cpu()), angle=angle_pred.cpu(), rotation_point='center', facecolor='blue', alpha=0.5, label=f'pred_mov_{i}'))

        patch_set.append(pch.Rectangle(pos.cpu() - size.cpu()/2, *(size.cpu()), angle=angle.cpu(), rotation_point='center', facecolor='red', label=f'true_mov_{i}'))
        patch_set.append(pch.Rectangle(pos_pred.cpu() - size_pred.cpu()/2, *(size_pred.cpu()), angle=angle_pred.cpu(), rotation_point='center', facecolor='blue', alpha=0.5, label=f'pred_mov_{i}'))
    return patch_set


def random_actions_policy(obs):
    num_envs, _ = obs.shape
    device = obs.device
    return 1.5 * torch.rand((num_envs, 3), device=device) - 0.75
    
def straight_policy(obs):
    num_envs, _ = obs.shape
    device = obs.device
    actions = torch.zeros((num_envs, 3), device=device)
    actions[:, 0] = 0.75
    return actions

def heuristic_policy(obs):
    pass

def teacher_policy(obs, policy):
    priv_obs_pred_all, actions = policy(obs)
    return priv_obs_pred_all, actions

def student_policy(obs, policy):
    priv_obs_pred_all, actions = policy(obs)
    return priv_obs_pred_all, actions

def full_teacher_policy(obs, policy):
    priv_obs_pred_all, actions = policy(obs)
    return priv_obs_pred_all, actions

if __name__ == "__main__":
    from tqdm import tqdm
    from pathlib import Path
    from high_level_policy import HLP_ROOT_DIR
    import glob
    import os
    from high_level_policy.envs.highlevelcontrol import HighLevelControlWrapper 
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_envs', type=int, default=2500,
                        help='the number of environments to run')
    
    parser.add_argument('--head', action='store_false', default=True,
                        help='run in headless mode')

    args = parser.parse_args()

    print(args.num_envs)
    print(args.head)


    num_envs = args.num_envs  
    env = HighLevelControlWrapper(num_envs=num_envs, headless=args.head, test=True)

    # recent_runs = sorted(glob.glob(f"{HLP_ROOT_DIR}/high_level_policy/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)
    model_path = EVAL_MODEL_PATH
    logger.configure(Path(model_path).resolve())
    policy_expert, policy_student  = load_env(headless=False)
    plots_path = f"{EVAL_MODEL_PATH}/plots_eval"


    num_eval_steps = 1000
    obs = env.reset()

    import torch

    # for i in tqdm(range(num_eval_steps)):
    #     with torch.no_grad():
    #         _, actions = policy_expert(obs)
    #     obs, rew, done, info = env.step(actions)
    # print(info['train/success'], info['train/failure'])
    # print(info['train/success_rate'])
    # print(info['train/ep_length']/info['train/env_count'])
    # obs = env.reset()
    patches = []
    for i in tqdm(range(num_eval_steps)):
        obs_truth = obs['obs'][0]
        priv_obs_truth = obs['privileged_obs'][0]
        with torch.no_grad():
            priv_obs_pred_all, actions = policy_student(obs)
        priv_obs_pred = priv_obs_pred_all[0]
        # patch_set = get_patch_set(obs_truth, priv_obs_truth, priv_obs_pred)
        # patches.append(patch_set)
        obs, rew, done, info = env.step(actions)
        # if done[0]:
        #     if not os.path.exists(plots_path):
        #         os.makedirs(plots_path)
        #     with open(f"{plots_path}/{i}.pkl", 'wb') as f:
        #         pickle.dump(patches, f)
        #     patches = []

    # print(info['train/episode']['success'])
    print(info['train/success'], info['train/failure'])
    print(info['train/success_rate'])
    print(info['train/ep_length']/info['train/env_count'])