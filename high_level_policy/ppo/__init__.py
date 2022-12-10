0# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import time
from collections import deque

import torch
from ml_logger import logger
from params_proto import PrefixProto
import os
import copy
import wandb
import numpy as np

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage

from high_level_policy import *

from matplotlib import pyplot as plt
from matplotlib import patches as pch
import matplotlib.animation as animation

FFwriter = animation.FFMpegWriter

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from high_level_policy.ppo.metrics_caches import DistCache, SlotCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'PPO'
    num_steps_per_env = 25  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 200
    save_anim_interval = 3
    log_freq = 10
    start_save_plot = 0
    save_plot_interval = 250
    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = '/home/dhruv/projects_dhruv/og_rlvrl/high_level_policy/runs/rapid-locomotion/2022-10-26/high_level_train/013418.804555/checkpoints/ac_weights_last.pt'  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt


class Runner:

    def __init__(self, env, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      ).to(self.device)

        if RunnerArgs.resume:
            print('loading walk model for init...')
            weights = logger.load_torch(RunnerArgs.checkpoint)
            # print(weights.keys())
            new_weights = {'.'.join(k.split('.')[1:]):v for k,v in weights.items() if k.startswith('critic')}
            # print(new_weights)
            actor_critic.critic_body.load_state_dict(state_dict=new_weights)
            # actor_critic.to(env.device)
            print('successfully loaded walk model...')


        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, eval_expert=False):
        from ml_logger import logger
        # initialize writer
        assert logger.prefix, "you will overwrite the entire instrument server"

        logger.start('start', 'epoch', 'episode', 'run', 'step')

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()

        print(privileged_obs.shape)

        rewbuffer = deque(maxlen=200)
        lenbuffer = deque(maxlen=200)
        rewbuffer_eval = deque(maxlen=200)
        lenbuffer_eval = deque(maxlen=200)
        patches = []
        patches_eval = []
        self.random_anim_env = 0 # np.random.randint(0, self.env.num_envs - self.env.num_train_envs)
        save_video_anim = False
        save_video_anim_eval = False
        complete_student = 1501

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if hasattr(self.env, "curriculum"):
            caches.__init__(curriculum_bins=len(self.env.curriculum))

        tot_iter = self.current_learning_iteration + num_learning_iterations
        save_at_iter = RunnerArgs.start_save_plot
        save_at_iter_eval = RunnerArgs.start_save_plot
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            # if self.current_learning_iteration+it % 5 == 0:
                # eval_expert = True
            
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    pos_rob, rot_rob = obs[self.random_anim_env, :2], obs[self.random_anim_env, 2:6]
                    angle_rob = torch.rad2deg(torch.atan2(2.0*(rot_rob[0]*rot_rob[1] + rot_rob[3]*rot_rob[2]), 1. - 2.*(rot_rob[1]*rot_rob[1] + rot_rob[2]*rot_rob[2])))

                    pos_rob_eval, rot_rob_eval = obs[num_train_envs, :2], obs[num_train_envs, 2:6]
                    angle_rob_eval = torch.rad2deg(torch.atan2(2.0*(rot_rob_eval[0]*rot_rob_eval[1] + rot_rob_eval[3]*rot_rob_eval[2]), 1. - 2.*(rot_rob_eval[1]*rot_rob_eval[1] + rot_rob_eval[2]*rot_rob_eval[2])))

                    
                    ### main policy calls
                    priv_train_pred, actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],obs_history[:num_train_envs], student=(it > complete_student))
                    
                    if eval_expert:
                        priv_obs_pred, actions_eval = self.alg.actor_critic.act_teacher(obs[num_train_envs:],
                                                                        privileged_obs[num_train_envs:])
                    else:
                        priv_obs_pred, actions_eval = self.alg.actor_critic.act_student(obs[num_train_envs:],
                                                                        obs_history[num_train_envs:])

                    ## 
                    
                    if USE_LATENT:
                        if it >= 0:
                            pos, pos_pred = privileged_obs[self.random_anim_env][:2], priv_train_pred[self.random_anim_env][:2]
                            rot, rot_pred = privileged_obs[self.random_anim_env][2:6], priv_train_pred[self.random_anim_env][2:6]
                            size, size_pred = privileged_obs[self.random_anim_env][6:8], priv_train_pred[self.random_anim_env][6:8]
                            weight, weight_pred = privileged_obs[self.random_anim_env][8], priv_train_pred[self.random_anim_env][8]

                            fpos, fpos_pred = privileged_obs[self.random_anim_env][9:11], priv_train_pred[self.random_anim_env][9:11]
                            fsize, fsize_pred = privileged_obs[self.random_anim_env][11:], priv_train_pred[self.random_anim_env][11:]

                            angle = torch.rad2deg(torch.atan2(2.0*(rot[0]*rot[1] + rot[3]*rot[2]), 1. - 2.*(rot[1]*rot[1] + rot[2]*rot[2])))
                            angle_pred = torch.rad2deg(torch.atan2(2.0*(rot_pred[0]*rot_pred[1] + rot_pred[3]*rot_pred[2]), 1. - 2.*(rot_pred[1]*rot_pred[1] + rot_pred[2]*rot_pred[2])))
                            
                            # print(fpos, fsize)

                            
                            # if round(angle.cpu().item(), 2) > 0.2:
                            #     print(rot, angle, rot_pred, angle_pred)
                            
                            patch_set = []

                            patch_set.append(pch.Rectangle(pos_rob.cpu().numpy() - np.array([0.588, 0.22]), width=0.588, height=0.22, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))
                            patch_set.append(pch.Rectangle(pos_rob.cpu().numpy() - np.array([0.588, 0.22]), width=0.588, height=0.22, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))
                            patch_set.append(pch.Rectangle(pos_rob.cpu().numpy() - np.array([0.588, 0.22]), width=0.588, height=0.22, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))

                            patch_set.append(pch.Rectangle(pos.cpu() - size.cpu()/2, *(size.cpu()), angle=angle.cpu(), rotation_point='center', facecolor='red', label='true_mov'))
                            patch_set.append(pch.Rectangle(pos_pred.cpu() - size_pred.cpu()/2, *(size_pred.cpu()), angle=angle_pred.cpu(), rotation_point='center', facecolor='blue', alpha=0.5, label='pred_mov'))
                            patch_set.append(pch.Rectangle(fpos.cpu() - fsize.cpu()/2, *(fsize.cpu()), alpha=1.0, facecolor='yellow', label='true_fixed'))
                            patch_set.append(pch.Rectangle(fpos_pred.cpu() - fsize_pred.cpu()/2, *(fsize_pred.cpu()), alpha=0.8, facecolor='black', label='pred_fixed'))

                            patch_set.append(pch.Rectangle(pos.cpu() - size.cpu()/2, *(size.cpu()), angle=angle.cpu(), rotation_point='center', facecolor='red', label='true_mov'))
                            patch_set.append(pch.Rectangle(pos_pred.cpu() - size_pred.cpu()/2, *(size_pred.cpu()), angle=angle_pred.cpu(), rotation_point='center', facecolor='blue', alpha=0.5, label='pred_mov'))

                            patch_set.append(pch.Rectangle(fpos.cpu() - fsize.cpu()/2, *(fsize.cpu()), alpha=1.0, facecolor='yellow', label='true_fixed'))
                            patch_set.append(pch.Rectangle(fpos_pred.cpu() - fsize_pred.cpu()/2, *(fsize_pred.cpu()), alpha=0.8, facecolor='black', label='pred_fixed'))
                            patch_set.append(weight)
                            patch_set.append(weight_pred)
                            patches.append(patch_set)
                        
                        
                        if it >= 0:
                            # patch_set.append(pch.Rectangle(pos_rob.cpu().numpy() - np.array([0.588, 0.22]), 0.588, 0.22, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))
                            # patch_set.append(pch.Rectangle(pos_rob.cpu().numpy() - np.array([0.588, 0.22]), 0.588, 0.22, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))
                            # patch_set.append(pch.Rectangle(pos_rob.cpu().numpy() - np.array([0.588, 0.22]), 0.588, 0.22, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))


                            pos_eval, pos_pred_eval = privileged_obs[num_train_envs][:2], priv_obs_pred[self.random_anim_env][:2]
                            rot_eval, rot_pred_eval = privileged_obs[num_train_envs][2:6], priv_obs_pred[self.random_anim_env][2:6]
                            size_eval, size_pred_eval = privileged_obs[num_train_envs][6:8], priv_obs_pred[self.random_anim_env][6:8]
                            weight_eval, weight_pred_eval = privileged_obs[num_train_envs][8], priv_train_pred[self.random_anim_env][8]

                            fpos_eval, fpos_pred_eval = privileged_obs[num_train_envs][9:11], priv_obs_pred[self.random_anim_env][9:11]
                            fsize_eval, fsize_pred_eval = privileged_obs[num_train_envs][11:], priv_obs_pred[self.random_anim_env][11:]

                            angle_eval = torch.rad2deg(torch.atan2(2.0*(rot_eval[0]*rot_eval[1] + rot_eval[3]*rot_eval[2]), 1. - 2.*(rot_eval[1]*rot_eval[1] + rot_eval[2]*rot_eval[2])))
                            angle_pred_eval = torch.rad2deg(torch.atan2(2.0*(rot_pred_eval[0]*rot_pred_eval[1] + rot_pred_eval[3]*rot_pred_eval[2]), 1. - 2.*(rot_pred_eval[1]*rot_pred_eval[1] + rot_pred_eval[2]*rot_pred_eval[2])))


                            patch_set_eval = []

                            patch_set_eval.append(pch.Rectangle(pos_rob_eval.cpu().numpy() - np.array([0.588, 0.22]), width=0.588, height=0.22, angle=angle_rob_eval.cpu(), rotation_point='center', facecolor='green', label='robot'))
                            patch_set_eval.append(pch.Rectangle(pos_rob_eval.cpu().numpy() - np.array([0.588, 0.22]), width=0.588, height=0.22, angle=angle_rob_eval.cpu(), rotation_point='center', facecolor='green', label='robot'))
                            patch_set_eval.append(pch.Rectangle(pos_rob_eval.cpu().numpy() - np.array([0.588, 0.22]), width=0.588, height=0.22, angle=angle_rob_eval.cpu(), rotation_point='center', facecolor='green', label='robot'))

                            patch_set_eval.append(pch.Rectangle(pos_eval.cpu() - size_eval.cpu()/2, *(size_eval.cpu()), angle=angle_eval.cpu(), rotation_point='center', facecolor='red', label='true_mov'))
                            patch_set_eval.append(pch.Rectangle(pos_pred_eval.cpu() - size_pred_eval.cpu()/2, *(size_pred_eval.cpu()), angle=angle_pred_eval.cpu(), rotation_point='center', facecolor='blue', alpha=0.5, label='pred_mov'))
                            patch_set_eval.append(pch.Rectangle(fpos_eval.cpu() - fsize_eval.cpu()/2, *(fsize_eval.cpu()), alpha=1.0, facecolor='yellow', label='true_fixed'))
                            patch_set_eval.append(pch.Rectangle(fpos_pred_eval.cpu() - fsize_pred_eval.cpu()/2, *(fsize_pred_eval.cpu()), alpha=0.8, facecolor='black', label='pred_fixed'))

                            patch_set_eval.append(pch.Rectangle(pos_eval.cpu() - size_eval.cpu()/2, *(size_eval.cpu()), angle=angle_eval.cpu(), rotation_point='center', facecolor='red', label='true_mov'))
                            patch_set_eval.append(pch.Rectangle(pos_pred_eval.cpu() - size_pred_eval.cpu()/2, *(size_pred_eval.cpu()), angle=angle_pred_eval.cpu(), rotation_point='center', facecolor='blue', alpha=0.5, label='pred_mov'))

                            patch_set_eval.append(pch.Rectangle(fpos_eval.cpu() - fsize_eval.cpu()/2, *(fsize_eval.cpu()), alpha=1.0, facecolor='yellow', label='true_fixed'))
                            patch_set_eval.append(pch.Rectangle(fpos_pred_eval.cpu() - fsize_pred_eval.cpu()/2, *(fsize_pred_eval.cpu()), alpha=0.8, facecolor='black', label='pred_fixed'))

                            patch_set_eval.append(weight_eval)
                            patch_set_eval.append(weight_pred_eval)

                            patches_eval.append(patch_set_eval)


                    # patch_set_eval.append(pch.Rectangle(pos_rob_eval.cpu().numpy() - np.array([0.588, 0.22]), 0.588, 0.22, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))
                    # patch_set_eval.append(pch.Rectangle(pos_rob_eval.cpu().numpy() - np.array([0.588, 0.22]), 0.588, 0.22, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))
                    # patch_set_eval.append(pch.Rectangle(pos_rob_eval.cpu().numpy() - np.array([0.588, 0.22]), 0.588, 0.22, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))

                    # fig = plt.figure(figsize=(8,8))
                    # ax = fig.gca()
                    # ax.set(xlim=(-2, 4), ylim=(-1, 1))
                    # ax.add_patch(pch.Rectangle(pos.cpu() - size.cpu()/2, *(size.cpu()), angle=angle.cpu(), rotation_point='center', facecolor='r', label='true_mov'))
                    # ax.add_patch(pch.Rectangle(pos_pred.cpu() - size_pred.cpu()/2, *(size_pred.cpu()), angle=angle_pred.cpu(), rotation_point='center', facecolor='b', alpha=0.5, label='pred_mov'))
                    # ax.add_patch(pch.Rectangle(fpos.cpu() - fsize.cpu()/2, *(fsize.cpu()), alpha=0.4, facecolor='brown', label='true_fixed'))
                    # ax.add_patch(pch.Rectangle(fpos_pred.cpu() - fsize_pred.cpu()/2, *(fsize_pred.cpu()), alpha=0.4, facecolor='black', label='pred_fixed'))
                    # ax.legend(loc='best')
                    

                    ######
                                            
                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))

                    obs_dict, rewards, dones, infos = ret
                    # print(dones[:5])
                    if dones[0]:
                        save_video_anim = True
                    if dones[num_train_envs]:
                        save_video_anim_eval = True

                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos:
                        # for k, v in infos['train/episode'].items():
                        #     wandb.log({f'train/{k}': v})
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])
                        

                    if 'eval/episode' in infos:
                        
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])

                    if 'curriculum' in infos:
                        curr_bins_train = infos['curriculum']['reset_train_env_bins']
                        curr_bins_eval = infos['curriculum']['reset_eval_env_bins']

                        caches.slot_cache.log(curr_bins_train, **{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('slot/train')
                        })
                        caches.slot_cache.log(curr_bins_eval, **{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('slot/eval')
                        })
                        caches.dist_cache.log(**{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('dist/train')
                        })
                        caches.dist_cache.log(**{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('dist/eval')
                        })

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                # Learning step
                self.alg.compute_returns(obs[:num_train_envs], privileged_obs[:num_train_envs], student=(it > complete_student), observation_history=obs_history[:num_train_envs])

                if it % eval_freq == 0:
                    self.env.reset_evaluation_envs()

                if it % eval_freq == 0:
                    logger.save_pkl({"iteration": it,
                                     **caches.slot_cache.get_summary(),
                                     **caches.dist_cache.get_summary()},
                                    path=f"curriculum/info.pkl", append=True)

            
            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_reconstruction_loss = self.alg.update(student=(it > complete_student))

            logger.store_metrics(
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                mean_reconstruction_loss=mean_reconstruction_loss
            )

            if USE_LATENT:

                if save_video_anim:
                    if it >= save_at_iter:
                        path = f'{HLP_ROOT_DIR}/tmp/legged_data'
                        os.makedirs(path, exist_ok=True)

                        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
                        last_patch = []
                        true_weight = ax[0].text(-0.8, 0.9, f"true:")
                        pred_weight = ax[0].text(-0.8, 0.8, f"predicted:")
                        # def init():
                        #     pass

                        def animate(frame):
                            # for txt in fig.texts:
                            #     txt.set_visible(False)
                            if len(last_patch) != 0:
                                for i in last_patch:
                                    i.remove()
                                last_patch.clear()
                            
                            robot, robot_1, robot_2 = frame[0], frame[1], frame[2]
                            true_mov, pred_mov, true_fix, pred_fix = frame[3], frame[4], frame[5], frame[6]
                            true_mov_copy, pred_mov_copy, true_fix_copy, pred_fix_copy = frame[7], frame[8], frame[9], frame[10]
                            
                            ax[0].add_patch(robot)
                            ax[0].add_patch(true_mov)
                            ax[0].add_patch(true_fix)
                            ax[0].add_patch(pred_mov)
                            ax[0].add_patch(pred_fix)
                            true_weight.set_text(f"true: {round(frame[11].item(), 3)}")
                            pred_weight.set_text(f"predicted: {round(frame[12].item(), 3)}")
                            ax[0].legend(loc='best')
                            ax[0].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='all')

                            ax[1].add_patch(robot_1)
                            ax[1].add_patch(true_mov_copy)
                            ax[1].add_patch(true_fix_copy)
                            ax[1].legend(loc='best')
                            ax[1].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='truth')

                            ax[2].add_patch(robot_2)
                            ax[2].add_patch(pred_mov_copy)
                            ax[2].add_patch(pred_fix_copy)
                            ax[2].legend(loc='best')
                            ax[2].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='predicted')

                            last_patch.extend(frame[:-2])    

                        tmp_img_path = f'{path}/{it}.mp4'

                        anim = animation.FuncAnimation(fig, animate, frames=patches, interval=10, repeat=False)
                        anim.save(tmp_img_path, writer = FFwriter(30))
                        plt.close()

                        # plt.savefig(tmp_img_path)
                        # plt.close()
                        logger.upload_file(file_path=tmp_img_path, target_path=f"plots/", once=False)
                        save_at_iter += RunnerArgs.save_plot_interval

                    patches = []
                    save_video_anim = False
                    self.random_anim_env = 0 # np.random.randint(0, self.env.num_envs - self.env.num_train_envs)
                    

                if save_video_anim_eval:
                    
                    if it >= save_at_iter_eval:
                        path = f'{HLP_ROOT_DIR}/tmp/legged_data'

                        os.makedirs(path, exist_ok=True)
                        
                        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
                        last_eval_patch = []
                        true_weight = ax[0].text(-0.8, 0.9, f"true:")
                        pred_weight = ax[0].text(-0.8, 0.8, f"predicted:")

                        # def init():
                        #     pass

                        def animate(frame):
                            # for txt in fig.texts:
                            #     txt.set_visible(False)

                            if len(last_eval_patch) != 0:
                                for i in last_eval_patch:
                                    i.remove()
                                last_eval_patch.clear()
                            
                            robot, robot_1, robot_2 = frame[0], frame[1], frame[2]
                            true_mov, pred_mov, true_fix, pred_fix = frame[3], frame[4], frame[5], frame[6]
                            true_mov_copy, pred_mov_copy, true_fix_copy, pred_fix_copy = frame[7], frame[8], frame[9], frame[10]
                            
                            ax[0].add_patch(robot)
                            ax[0].add_patch(true_mov)
                            ax[0].add_patch(true_fix)
                            ax[0].add_patch(pred_mov)
                            ax[0].add_patch(pred_fix)
                            true_weight.set_text(f"true: {round(frame[12].item(), 3)}")
                            pred_weight.set_text(f"predicted: {round(frame[12].item(), 3)}")
                            ax[0].legend(loc='best')
                            ax[0].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='all')

                            ax[1].add_patch(robot_1)
                            ax[1].add_patch(true_mov_copy)
                            ax[1].add_patch(true_fix_copy)
                            ax[1].legend(loc='best')
                            ax[1].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='truth')

                            ax[2].add_patch(robot_2)
                            ax[2].add_patch(pred_mov_copy)
                            ax[2].add_patch(pred_fix_copy)
                            ax[2].legend(loc='best')
                            ax[2].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='predicted')
                            
                            last_eval_patch.extend(frame[:-2])

                        tmp_img_path = f'{path}/{it}_eval.mp4'

                        anim = animation.FuncAnimation(fig, animate, frames=patches_eval, interval=10, repeat=False)
                        anim.save(tmp_img_path, writer = FFwriter(30))
                        plt.close()

                        # plt.savefig(tmp_img_path)
                        # plt.close()
                        logger.upload_file(file_path=tmp_img_path, target_path=f"plots_eval/", once=False)
                        save_at_iter_eval += RunnerArgs.save_plot_interval
                    patches_eval = []
                    save_video_anim_eval = False
                
            if RunnerArgs.save_video_interval:                
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
                # for k, v in self.extras['eval/episode'].items():
                #     wandb.log({f'eval/{k}': v})
                
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                print(logger.summary_caches[None])
                logger.job_running()

                

            if it % RunnerArgs.save_interval == 0:
                with logger.Sync():
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = f'{HLP_ROOT_DIR}/tmp/legged_data'

                    os.makedirs(path, exist_ok=True)

                    body_path = f'{path}/body_latest.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(body_path)
                    logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

                    

                    if USE_LATENT:
                        adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                        adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                        traced_script_adaptation_module = torch.jit.script(adaptation_module)
                        traced_script_adaptation_module.save(adaptation_module_path)
                        logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)

                    

            self.current_learning_iteration += num_learning_iterations  

        with logger.Sync():
            logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            path = f'{HLP_ROOT_DIR}/tmp/legged_data'

            os.makedirs(path, exist_ok=True)

            body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

            if USE_LATENT:
                adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                traced_script_adaptation_module = torch.jit.script(adaptation_module)
                traced_script_adaptation_module.save(adaptation_module_path)
                logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)

    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            # self.env.ll_env.start_recording()
            # if self.env.num_eval_envs > 0:
            self.env.ll_env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        # frames = self.env.ll_env.get_complete_frames()
        # if len(frames) > 0:
        #     self.env.ll_env.pause_recording()
        #     print("LOGGING VIDEO")
        #     logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.ll_env.dt)

        # if self.env.num_eval_envs > 0:
        frames = self.env.ll_env.get_complete_frames_eval()
        if len(frames) > 0:
            self.env.ll_env.pause_recording_eval()
            print("LOGGING EVAL VIDEO")
            logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.ll_env.dt)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
