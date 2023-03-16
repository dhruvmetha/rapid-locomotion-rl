from isaacgym import gymapi
from PIL import Image
from ml_logger import logger
from high_level_policy import *

import numpy as np
# from matplotlib import pyplot as plt
from matplotlib import patches as pch
# import matplotlib.animation as animation
import pickle
import argparse
import threading
import queue
from datetime import datetime

class Worker(threading.Thread):
    def __init__(self, queue, fn):
        threading.Thread.__init__(self)
        self.queue = queue
        self.data_path = Path(f'/common/users/dm1487/legged_manipulation_data/rollout_data/{SAVE_ADAPTATION_DATA_FILE_NAME}')
        # make directory `data_path` if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.print = fn

    def run(self):
        while True:
            # get the next item from the queue
            # print('heree')
            item = self.queue.get()
            
            # self.lock.acquire()
            # try: 
            #     self.print('here')
            #     self.print(len(item))
            # finally:
            #     self.lock.release()

            # save the data
            for k, v in item.items():
                item[k] = torch.cat(v, dim=0)
                # print(item[k].shape)
                if isinstance(item[k], torch.Tensor):
                    item[k] = item[k].cpu().numpy()
            np.savez_compressed(self.data_path/f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz', **item)
            
            # with open(SAVE_ADAPTATION_DATA_FILE_NAME, 'wb') as f:
            #     pickle.dump(item, f)

            # send a signal to the queue that the job is done
            self.queue.task_done()

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
        num_obs_history=(env.num_obs+24) * \
                        ROLLOUT_HISTORY,
        num_actions=env.num_actions)

    # print(actor_critic)

    # print(logger.prefix)
    # print(logger.glob("*"))
    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    print('here actor_critic')
    policy_student = actor_critic.act_inference
    print('here policy_student')
    policy_expert = actor_critic.act_inference_expert
    print('here policy_expert')
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
    
def straight_motion_policy(obs):
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

def heuristic_policy_1(prev_obs, obs):
    actions[:, 0] = 0.75
    actions[:, 1] = 0.
    actions[:, 2] = 0.
    if torch.linalg.norm(obs[:, 0] - prev_obs[:, 0]) < 0.1:
        actions[:, 0] = 0
        if np.random.uniform(0, 1) > 0.5:
            actions[:, 1] = 0.5
        else:
            actions[:, 1] = -0.5
    
    if torch.linalg.norm(obs[:, 1] - prev_obs[:, 1]) < 0.1:
        actions[:, 0] = 0.5
        actions[:, 1] = 0
    
    return actions

def create_env(num_envs, headless, full_info, train_ratio=0.0):
    from high_level_policy.envs.highlevelcontrol import HighLevelControlWrapper
    env = HighLevelControlWrapper(num_envs=num_envs, headless=headless, test=True, full_info=full_info, train_ratio=train_ratio)
    return env

if __name__ == "__main__":
    from tqdm import tqdm
    from pathlib import Path
    from high_level_policy import HLP_ROOT_DIR
    import glob
    import os
    from high_level_policy.envs.highlevelcontrol import HighLevelControlWrapper 
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_envs', type=int, default=4096,
                        help='the number of environments to run')
    
    parser.add_argument('--iter', type=int, default=10000,
                        help='the number of environments to run')
    
    parser.add_argument('--head', action='store_true', default=False,
                        help='run in headless mode')

    args = parser.parse_args()

    num_envs = args.num_envs
    iters = args.iter
    headless = not args.head
    print(args.num_envs)
    print(headless)
    all_modes = ["random_motion_policy", "straight_motion_policy", "heuristic_1", "heuristic_2", "main_policy", "teacher_policy", "full_teacher_policy"]
    mode = "teacher_policy"

    if mode == "full_teacher_policy":
        env = create_env(num_envs, headless, full_info=True, train_ratio=0.0)
        action_func = full_teacher_policy
    elif mode == "straight_motion_policy":
        env = create_env(num_envs, headless, full_info=True, train_ratio=0.0)
        action_func = straight_motion_policy
    elif mode == "random_motion_policy":
        env = create_env(num_envs, headless, full_info=True, train_ratio=0.0)
        action_func = random_actions_policy
    elif mode == "teacher_policy":
        env = create_env(num_envs, headless, full_info=False, train_ratio=0.99)
        action_func = teacher_policy
    elif mode == "main_policy":
        env = create_env(num_envs, headless, full_info=False, train_ratio=0.0)
        action_func = student_policy
    

      
    # env = HighLevelControlWrapper(num_envs=num_envs, headless=head, test=True)
    

    recent_runs = sorted(glob.glob(f"{HLP_ROOT_DIR}/high_level_policy/runs/{task_inplay}/*/*/*"), key=os.path.getmtime)
    model_path = recent_runs[-1]
    # model_path = "/common/home/dm1487/robotics_research/legged_manipulation/experiments_real_robot/high_level_policy/runs/task_pure_rl/2023-02-18/high_level_train/233250.669431"
    # model_path = "/common/home/dm1487/robotics_research/legged_manipulation/experiments_real_robot/high_level_policy/runs/task_pure_rl/2023-02-21/high_level_train/103519.121655"
    # model_path = "/common/home/dm1487/robotics_research/legged_manipulation/experiments_real_robot/high_level_policy/runs/task_pure_rl/2023-02-21/high_level_train/210610.485965"
    # model_path = "/common/home/dm1487/robotics_research/legged_manipulation/experiments_real_robot/high_level_policy/runs/task_pure_rl/2023-02-23/high_level_train/021407.085820"
    # model_path = "/common/home/dm1487/robotics_research/legged_manipulation/experiments_real_robot/high_level_policy/runs/task_pure_rl/2023-02-23/high_level_train/222927.684309"
    # model_path = "/common/home/dm1487/robotics_research/legged_manipulation/experiments_real_robot/high_level_policy/runs/task_pure_rl/2023-02-26/high_level_train/222315.629972"
    # model_path = "/common/home/dm1487/robotics_research/legged_manipulation/experiments_real_robot/high_level_policy/runs/task_pure_rl/2023-02-26/high_level_train/195203.010990"
    # model_path = "/common/home/dm1487/robotics_research/legged_manipulation/experiments_real_robot/high_level_policy/runs/task_teacher_decoder/2023-02-27/high_level_train/190152.158246"
    print("#################")
    print(model_path)
    logger.configure(Path(model_path).resolve())

    print("#################")

    print('configuring logger')

    policy_expert, policy_student  = load_env(headless=headless)
    plots_path = f"{model_path}/plots_eval"

    num_eval_steps = iters
    obs = env.reset()

    # print(obs)
    import torch

    if SAVE_ADAPTATION_DATA:

        q = queue.Queue(maxsize=200)
        worker = Worker(q, fn=print)
        worker.start()
        q.join()

    patches = []
    data = {
        "obs_data" : [],
        "priv_obs_data" : [],
        "obs_hist_data" : [],
        "fsw_data" : [],
        "done_data" : []
    }

    print(env.ll_env.full_seen_world.shape)
    

    env.ll_env.start_recording()
    for i in tqdm(range(num_eval_steps)):
        # obs_truth = obs['obs'][0] 
        # priv_obs_truth = obs['privileged_obs'][0]
        # print(obs['privileged_obs'][0])

        

        with torch.no_grad():
            priv_obs_pred_all, actions = policy_expert(obs)
        # priv_obs_pred = priv_obs_pred_all[0]
        # patch_set = get_patch_set(obs_truth, priv_obs_truth, priv_obs_pred)
        # patches.append(patch_set)
        obs, rew, done, info = env.step(actions)
        
        bx = env.ll_env.env_origins[env.ll_env.num_envs -1][0] # obs['obs'][-1, 0]
        by = env.ll_env.env_origins[env.ll_env.num_envs -1][1] # obs['obs'][-1, 1]
        bz = 0.0

        env.ll_env.gym.set_camera_location(env.ll_env.rendering_camera_eval, env.ll_env.envs[env.ll_env.num_envs-1],
                                            gymapi.Vec3(bx+1.0, by, bz + 3.2),
                                            gymapi.Vec3(bx+1.2, by, 0.5))
        frame = env.ll_env.gym.get_camera_image(env.ll_env.sim, env.ll_env.envs[env.ll_env.num_envs-1],
                                                                env.ll_env.rendering_camera_eval,
                                                                gymapi.IMAGE_COLOR)

        # print(frame.shape)
        
        # img = Image.fromarray(frame).resize((368, 240))
        # logger.save_image(img, f"env_images/test_sample/{i:05d}.png")

        # env.reset()
        
        # plt.imshow(frame)
        # plt.show()

        if done[0]:
            frames = env.ll_env.get_complete_frames()
            print("done", len(frames))
            if len(frames) > 0:
                env.ll_env.pause_recording()
                logger.save_video(frames, f"play_videos/{i:05d}.mp4", fps=1 / env.ll_env.dt)
                env.ll_env.start_recording()

        # if done[0]:
        #     print(obs['obs'][0])
        #     print(obs['privileged_obs'][0])
        #     print(env.ll_env.full_seen_world.unsqueeze(0).clone()[0, 0])
        #     print(obs['obs_history'][0, -36:-34])
        
        if SAVE_ADAPTATION_DATA:

            data['obs_data'].append(obs['obs'].unsqueeze(0).clone())
            data['priv_obs_data'].append(obs['privileged_obs'].unsqueeze(0).clone())
            data['obs_hist_data'].append(obs['obs_history'][:, -36:].unsqueeze(0).clone())
            data['fsw_data'].append(env.ll_env.full_seen_world.unsqueeze(0).clone())
            data['done_data'].append(done.view(-1, 1).unsqueeze(0).clone())

            if i == 0:
                print(data['fsw_data'][-1].shape)
                print(data['priv_obs_data'][-1].shape)
                print(data['obs_hist_data'][-1].shape)
                print(data['obs_data'][-1].shape)

            if (i+1)%25 == 0:
                q.put(data)
                data = {
                    "obs_data" : [],
                    "priv_obs_data" : [],
                    "obs_hist_data" : [],
                    "fsw_data" : [],
                    "done_data" : []
                }

        if done.nonzero().size(0) > 0:
            env.reset_obs_history(done.nonzero())
        # print(env.ll_env.full_seen_world.shape)
        # print(env.ll_env.full_seen_world[0])
        # print(env.ll_env.full_seen_world.shape)
        # print(obs['obs'][0])
        
        # print(obs['obs'][0, 0])
        # print((obs['obs'][:, 0] > 3.5).nonzero())

        # if done[0]:
        #     if not os.path.exists(plots_path):
        #         os.makedirs(plots_path)
        #     with open(f"{plots_path}/{i}.pkl", 'wb') as f:
        #         pickle.dump(patches, f)
        #     patches = []

    # print(info['train/episode']['success'])
    # print(info['train/episode']['success'], info['train/episode']['failure'])
    print(info)
    if 'success_rate' in info['train/episode']:
        print('train success rate', info['train/episode']['success_rate'])
    if 'success_rate' in info['eval/episode']:
        print('eval success rate', info['eval/episode']['success_rate'])

    # print(info['train/episode']['ep_length']/info['train/episode']['env_count'])


