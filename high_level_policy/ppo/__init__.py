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
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import patches as pch
import matplotlib.animation as animation
import pickle

FFwriter = animation.FFMpegWriter

np.random.seed(42)
torch.manual_seed(42)

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
    save_interval = 50  # check for potential saves every this many iterations
    save_video_interval = 40
    save_anim_interval = 3
    log_freq = 10   
    start_save_plot = 0
    save_plot_interval = 125
    # load and resume
    resume = True
    load_run = -1  # -1 = last run
    checkpoint = '/common/home/dm1487/robotics_research/legged_manipulation/experimental_bed_2/high_level_policy/runs/task_full_info_decoder/2023-02-14/high_level_train/183329.086403/checkpoints/ac_weights_last.pt'  # -1 = last saved model
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
            print('loading model for init...')
            weights = logger.load_torch(RunnerArgs.checkpoint)
            # print(weights.keys())
            actor_critic.load_state_dict(state_dict=weights)
            # new_weights = {'.'.join(k.split('.')[1:]):v for k,v in weights.items() if k.startswith('critic')}
            # # print(new_weights)
            # actor_critic.critic_body.load_state_dict(state_dict=new_weights)
            # actor_critic.to(env.device)
            print('successfully loaded model...')


        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env


        if ENCODER:
            latent_size = LATENT_DIM_SIZE
        else:
            latent_size = PER_RECT * RECTS

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions], [HIDDEN_STATE_SIZE], [latent_size])

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

        adaptation_hidden_states = torch.zeros(self.env.num_envs, HIDDEN_STATE_SIZE).to(self.device)
        latent_size = LATENT_DIM_SIZE if (USE_LATENT and ENCODER) else PER_RECT * RECTS 
        latent_teacher_state = torch.zeros(self.env.num_envs, latent_size).to(self.device)
        
        self.alg.actor_critic.train()

        print(privileged_obs.shape)

        rewbuffer = deque(maxlen=200)
        lenbuffer = deque(maxlen=200)
        rewbuffer_eval = deque(maxlen=200)
        lenbuffer_eval = deque(maxlen=200)
        patches = []
        patches_eval = []
        # self.random_anim_env = np.random.choice(np.arange(0, num_train_envs))
        self.random_anim_env = 0 # np.random.randint(0, self.env.num_envs - self.env.num_train_envs)
        self.random_eval_anim_env = -1 # np.random.randint(self.env.num_train_envs, self.env.num_envs)
        self.eval_dones_ctr = 0 # np.random.randint(0, self.env.num_envs - self.env.num_train_envs)
        self.num_patches = 0
        save_video_anim = False
        save_video_anim_eval = False
        complete_student = STUDENT_ENCODING
        complete_student_ctr = 0 

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if hasattr(self.env, "curriculum"):
            caches.__init__(curriculum_bins=len(self.env.curriculum))

        tot_iter = self.current_learning_iteration + num_learning_iterations
        save_at_iter = RunnerArgs.start_save_plot
        save_at_iter_eval = RunnerArgs.start_save_plot
        for it in tqdm(range(self.current_learning_iteration, tot_iter)):
            self.env.ll_env.world_asset.variables['full_info'] = False # it < 100
            start = time.time()
            # Rollout
            # if self.current_learning_iteration+it % 5 == 0:
                # eval_expert = True
            
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    full_seen_world = self.env.full_seen_world

                    pos_rob, rot_rob = obs[self.random_anim_env, :2], obs[self.random_anim_env, 2:6]
                    angle_rob = torch.rad2deg(torch.atan2(2.0*(rot_rob[0]*rot_rob[1] + rot_rob[3]*rot_rob[2]), 1. - 2.*(rot_rob[1]*rot_rob[1] + rot_rob[2]*rot_rob[2])))

                    pos_rob_eval, rot_rob_eval = obs[self.random_eval_anim_env, :2], obs[self.random_eval_anim_env, 2:6]
                    angle_rob_eval = torch.rad2deg(torch.atan2(2.0*(rot_rob_eval[0]*rot_rob_eval[1] + rot_rob_eval[3]*rot_rob_eval[2]), 1. - 2.*(rot_rob_eval[1]*rot_rob_eval[1] + rot_rob_eval[2]*rot_rob_eval[2])))

                    
                    ### main policy calls
                    ((priv_train_pred_teacher, priv_train_pred_student), (latent_enc_teacher, latent_enc_student), next_hidden_states), actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs], obs_history[:num_train_envs], adaptation_hidden_states[:num_train_envs, :], latent_teacher_state[:num_train_envs, :], full_seen_world[:num_train_envs, :], rollout=True, student=(it > complete_student))
                    
                    priv_train_pred = priv_train_pred_teacher
                    latent_enc = latent_enc_teacher

                    next_hidden_states_eval = None
                    
                    if eval_expert:
                        (priv_obs_pred, latent_pred), actions_eval = self.alg.actor_critic.act_teacher(obs[num_train_envs:], privileged_obs[num_train_envs:])
                    else:
                        (priv_obs_pred, latent_pred, next_hidden_states_eval), actions_eval = self.alg.actor_critic.act_student(obs[num_train_envs:], obs_history[num_train_envs:], adaptation_hidden_states[num_train_envs:, :])

                    
                    if USE_LATENT and DECODER and CREATE_VIZ:
                        start = time.time()
                        if it >= 0:

                            patch_set = []

                            for _ in range(4):
                                patch_set.append(pch.Rectangle(pos_rob_eval.cpu().numpy() - np.array([0.588/2, 0.22/2]), width=0.588, height=0.22, angle=angle_rob_eval.cpu(), rotation_point='center', facecolor='green', label='robot'))
                            
                            for i in range(RECTS):
                                j = i*PER_RECT + 2

                                pos, pos_pred, pos_fsw = privileged_obs[self.random_eval_anim_env][j:j+2], priv_obs_pred[self.random_eval_anim_env][j:j+2], full_seen_world[self.random_eval_anim_env][j:j+2]

                                angle, angle_pred, angle_fsw = torch.rad2deg(privileged_obs[self.random_eval_anim_env][j+2:j+3]), torch.rad2deg(priv_obs_pred[self.random_eval_anim_env][j+2:j+3]), full_seen_world[self.random_eval_anim_env][j+2:j+3]
                                
                                size, size_pred, size_fsw = privileged_obs[self.random_eval_anim_env][j+3:j+5], priv_obs_pred[self.random_eval_anim_env][j+3:j+5], full_seen_world[self.random_eval_anim_env][j+3:j+5]

                                block_color = 'red'
                                if privileged_obs[self.random_eval_anim_env][j-1] == 1:
                                    block_color = 'yellow'

                                fsw_block_color = 'red'
                                if full_seen_world[self.random_eval_anim_env][j-1] == 1:
                                    fsw_block_color = 'yellow'
                                
                                pred_block_color = 'blue'
                                if priv_obs_pred[self.random_eval_anim_env][j-1] > 0.8:
                                    pred_block_color = 'orange'

                                for _ in range(2):
                                    patch_set.append(pch.Rectangle(pos.cpu() - size.cpu()/2, *(size.cpu()), angle=angle.cpu(), rotation_point='center', facecolor=block_color, label=f'true_mov_{i}'))
                                    
                                    patch_set.append(pch.Rectangle(pos_pred.cpu() - size_pred.cpu()/2, *(size_pred.cpu()), angle=angle_pred.cpu(), rotation_point='center', facecolor=pred_block_color, alpha=0.5, label=f'pred_mov_{i}'))

                                    patch_set.append(pch.Rectangle(pos_fsw.cpu() - size_fsw.cpu()/2, *(size_fsw.cpu()), angle=angle_fsw.cpu(), rotation_point='center', facecolor=fsw_block_color, label=f'seen_mov_{i}'))
                            
                            patches.append(patch_set)


                            patch_set_eval = []

                            for _ in range(4):
                                patch_set_eval.append(pch.Rectangle(pos_rob.cpu().numpy() - np.array([0.588/2, 0.22/2]), 0.588, 0.22, angle=angle_rob.cpu(), rotation_point='center', facecolor='green', label='robot'))
                            
                            for i in range(RECTS):
                                j = i*PER_RECT + 2

                                pos, pos_pred, pos_fsw = privileged_obs[self.random_anim_env][j:j+2], priv_train_pred[self.random_anim_env][j:j+2], full_seen_world[self.random_anim_env][j:j+2]
                                angle, angle_pred, angle_fsw = torch.rad2deg(privileged_obs[self.random_anim_env][j+2:j+3]), torch.rad2deg(priv_train_pred[self.random_anim_env][j+2:j+3]), torch.rad2deg(full_seen_world[self.random_anim_env][j+2:j+3])
                                size, size_pred, size_fsw = privileged_obs[self.random_anim_env][j+3:j+5], priv_train_pred[self.random_anim_env][j+3:j+5], full_seen_world[self.random_anim_env][j+3:j+5]
                                

                                block_color = 'red'
                                if privileged_obs[self.random_anim_env][j-1] == 1:
                                    block_color = 'yellow'

                                fsw_block_color = 'red'
                                if full_seen_world[self.random_anim_env][j-1] == 1:
                                    fsw_block_color = 'yellow'
                                
                                pred_block_color = 'blue'
                                if priv_train_pred[self.random_anim_env][j-1] > 0.8:
                                    pred_block_color = 'orange'

                                
                                for _ in range(2):
                                    patch_set_eval.append(pch.Rectangle(pos.cpu() - size.cpu()/2, *(size.cpu()), angle=angle.cpu(), rotation_point='center', facecolor=block_color, label=f'true_mov_{i}'))
                                    
                                    patch_set_eval.append(pch.Rectangle(pos_pred.cpu() - size_pred.cpu()/2, *(size_pred.cpu()), angle=angle_pred.cpu(), rotation_point='center', facecolor=pred_block_color, alpha=0.5, label=f'pred_mov_{i}'))

                                    patch_set_eval.append(pch.Rectangle(pos_fsw.cpu() - size_fsw.cpu()/2, *(size_fsw.cpu()), angle=angle_fsw.cpu(), rotation_point='center', facecolor=fsw_block_color, label=f'seen_mov_{i}'))
                            patches_eval.append(patch_set_eval)
                        # print('patch_creation', time.time() - start)

                    ######
                                            
                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))

                    obs_dict, rewards, dones, infos = ret
                    # print(dones[:5])
                    if dones[self.random_anim_env]:
                        save_video_anim = True
                    if dones[self.random_eval_anim_env]:
                        self.eval_dones_ctr += 1
                        if self.eval_dones_ctr >= 2:
                            save_video_anim_eval = True
                            self.eval_dones_ctr = 0

                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    # print(obs_history.shape, latent_enc.shape, latent_pred.shape, obs_history[0, -20:])

                    
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)
                    
                    if next_hidden_states is not None:
                        adaptation_hidden_states[:num_train_envs, :] = next_hidden_states
                    if next_hidden_states_eval is not None:
                        adaptation_hidden_states[num_train_envs:, :] = next_hidden_states_eval

                    adaptation_hidden_states[dones, :] = 0.

                    if False and USE_LATENT:
                        if (it > TEACHER_FORCING):
                            obs_history[:num_train_envs, -20:] = latent_enc_student[:]
                        else:
                            obs_history[:num_train_envs, -20:] = latent_enc_teacher[:]
                        obs_history[num_train_envs:, -20:] = latent_pred[:]
                        # print('internal', obs_history[0, -25:])

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
                self.alg.compute_returns(obs[:num_train_envs], privileged_obs[:num_train_envs], student=((it > complete_student)), observation_history=obs_history[:num_train_envs])

                # if it % eval_freq == 0:
                #     self.env.reset_evaluation_envs()

                if it % eval_freq == 0:
                    logger.save_pkl({"iteration": it,
                                     **caches.slot_cache.get_summary(),
                                     **caches.dist_cache.get_summary()},
                                    path=f"curriculum/info.pkl", append=True)

            
            mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_kl, mean_adaptation_module_loss, mean_reconstruction_loss, mean_adaptation_reconstruction_loss = self.alg.update(student=((it > complete_student)))

            # mean_eval_adaptation_module_loss = 0
            # mean_eval_teacher_reconstruction_loss = 0
            # mean_eval_adaptation_reconstruction_loss = 0

            # if USE_LATENT:
            #     with torch.inference_mode():
            #         (priv_obs_pred_teacher, latent_pred_teacher), _ = self.alg.actor_critic.act_teacher(obs[num_train_envs:], privileged_obs[num_train_envs:])

            #         (priv_obs_pred_student, latent_pred_student), _ = self.alg.actor_critic.act_student(obs[num_train_envs:], obs_history[num_train_envs:])

            #         mean_eval_adaptation_module_loss = torch.nn.functional.mse_loss(latent_pred_teacher, latent_pred_student).item()

            #         if DECODER:
            #             mean_eval_teacher_reconstruction_loss = torch.nn.functional.mse_loss(privileged_obs[num_train_envs:], priv_obs_pred_teacher).item()
            #             mean_eval_adaptation_reconstruction_loss = torch.nn.functional.mse_loss(privileged_obs[num_train_envs:], priv_obs_pred_student).item()

            logger.store_metrics(
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                mean_reconstruction_loss=mean_reconstruction_loss,
                mean_entropy_loss=mean_entropy_loss,
                mean_kl=mean_kl,
                mean_adaptation_reconstruction_loss=mean_adaptation_reconstruction_loss,
                # mean_eval_teacher_reconstruction_loss = mean_eval_teacher_reconstruction_loss,
                # mean_eval_adaptation_module_loss = mean_eval_adaptation_module_loss,
                # mean_eval_adaptation_reconstruction_loss = mean_eval_adaptation_reconstruction_loss,
            )

            if USE_LATENT and DECODER and CREATE_VIZ:
                

                if save_video_anim_eval:
                    start = time.time()
                    try:
                        path = f'{HLP_ROOT_DIR}/tmp/legged_data_{task_inplay}'
                        # if not os.path.exists(path):
                        os.makedirs(path, exist_ok=True)
                        tmp_img_path = f'{path}/{it}.pkl'
                        with open(tmp_img_path, 'wb') as f:
                            pickle.dump(patches, f)
                        
                        logger.upload_file(file_path=tmp_img_path, target_path=f"plots_eval/", once=False)
                        os.remove(tmp_img_path)
                    except:
                        pass

                    save_at_iter += RunnerArgs.save_plot_interval

                    patches = []
                    save_video_anim_eval = False
                    # print(f"Saving eval patch took {time.time() - start} seconds")
                    # self.random_eval_anim_env = -1 # np.random.randint(self.env.num_train_envs, self.env.num_envs)

                if save_video_anim:
                    start = time.time()
                    try:
                        path = f'{HLP_ROOT_DIR}/tmp/legged_data_{task_inplay}_eval/'
                        os.makedirs(path, exist_ok=True)
                        tmp_img_path = f'{path}/{it}.pkl'
                        with open(tmp_img_path, 'wb') as f:
                            pickle.dump(patches_eval, f)
                        logger.upload_file(file_path=tmp_img_path, target_path=f"plots/", once=False)
                        os.remove(tmp_img_path)
                    except:
                        pass
                    patches_eval = []
                    save_video_anim = False
                    # print(f"Saving patch took {time.time() - start} seconds")
                    # self.random_anim_env = 0 # np.random.randint(0, self.env.num_envs - self.env.num_train_envs)
                
            if RunnerArgs.save_video_interval:                
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
                # for k, v in self.extras['eval/episode'].items():
                #     wandb.log({f'eval/{k}': v})
                
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                print(logger.summary_caches[None])
                print(self.env.world_dist)
                print(self.env.world_ctr)
                print(self.env.world_success)
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
                        if not LSTM_ADAPTATION:
                            traced_script_adaptation_module = torch.jit.script(adaptation_module)
                        else:
                            x = torch.randn(1, ROLLOUT_HISTORY, 37, device='cpu')
                            hidden = torch.zeros(1, 1, HIDDEN_STATE_SIZE,  device='cpu')
                            traced_script_adaptation_module = torch.jit.trace(adaptation_module, (x, hidden))
                        
                        traced_script_adaptation_module.save(adaptation_module_path)
                        logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)

            self.env.increment_training_ctr()
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
            self.env.ll_env.start_recording()
            # if self.env.num_eval_envs > 0:
            self.env.ll_env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.ll_env.get_complete_frames()
        if len(frames) > 0:
            self.env.ll_env.pause_recording()
            print("LOGGING VIDEO")
            logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.ll_env.dt)

        # if self.env.num_eval_envs > 0:
        frames = self.env.ll_env.get_complete_frames_eval()
        if len(frames) > 0:
            self.env.ll_env.pause_recording_eval()
            print("LOGGING EVAL VIDEO")
            logger.save_video(frames, f"videos_eval/{it:05d}_eval.mp4", fps=1 / self.env.ll_env.dt)

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
