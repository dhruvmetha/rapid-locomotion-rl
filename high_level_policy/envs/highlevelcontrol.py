import isaacgym
assert isaacgym
import torch
import numpy as np
import os
import wandb

from mini_gym.envs import *
from mini_gym.envs.base.legged_robot_config import Cfg
from mini_gym.envs.go1.go1_config import config_go1
from mini_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from high_level_policy import *

if not os.path.exists(TRAJ_IMAGE_FOLDER):
    os.makedirs(TRAJ_IMAGE_FOLDER)

class HighLevelControlWrapper():
    def __init__(self, num_envs=1, headless=False, test=False):
        self.device= 'cuda:0'
        if world_cfg.CUSTOM_BLOCK:
            self.num_obs = 24 if world_cfg.fixed_block.add_to_obs else 22
        else:
            self.num_obs = 13
        self.num_actions = 3
        self.max_episode_length_s = 5 if test else 10 # 12
        self.num_privileged_obs = 18
        self.num_obs_history = 16

        self.num_envs = num_envs
        self.num_train_envs = max(1, int(num_envs*0.95))

        self.ll_env, self.low_level_policy = self._load_env(num_envs, headless)
        self.dt = self.ll_env.dt
        self.max_episode_length = int(self.max_episode_length_s / self.ll_env.dt)
        self.ll_env.commands[:, :3] = 0.
        self.ll_obs = self.ll_env.reset()
        self.ll_rew = self.ll_env.rew_buf

        self.all_trajectories = []
        self.all_rects = []
        self.success_trajectories = []

        self.traj_id = torch.tensor(np.random.uniform(self.num_train_envs, self.num_envs), device=self.device, dtype=torch.int, requires_grad=False)
        self.traj_image_id = 0

        self.obs_buf = torch.zeros((num_envs, self.num_obs), device=self.device, dtype=torch.float, requires_grad=False)
        self.rew_buf = torch.zeros(num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.discount_rew_buf = torch.zeros(num_envs, device=self.device, dtype=torch.float, requires_grad=False) + 1.0
        self.gs_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.reset_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.time_buf = torch.zeros(num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.int, requires_grad=False)
        self.actions = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=torch.float, requires_grad=False)
        self.last_actions = torch.zeros((num_envs, self.num_actions), device=self.device, dtype=torch.float, requires_grad=False)

        self.last_pos = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)
        self.last_pos = self.ll_env.root_states[:, :3] - self.ll_env.env_origins[:, :3] - self.ll_env.base_init_state[:3]
        self.dist_travelled = torch.zeros(num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.lateral_vel = torch.zeros(num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.backward_vel = torch.zeros(num_envs, device=self.device, dtype=torch.float, requires_grad=False)
        self.angular_vel = torch.zeros(num_envs, device=self.device, dtype=torch.float, requires_grad=False)

        self.privileged_obs_buf = torch.zeros((num_envs, self.num_privileged_obs), device=self.device, dtype=torch.float, requires_grad=False)
        self.obs_history = torch.zeros(self.num_envs, self.num_obs_history, dtype=torch.float, device=self.device, requires_grad=False)
        self.trajectory = torch.zeros(self.max_episode_length+1, 2, dtype=torch.float, device=self.device, requires_grad=False)

        self.goal_position = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device, requires_grad=False)
        # self.goal_position = (1.5 * torch.rand((self.num_envs, 2), dtype=torch.float, device=self.device, requires_grad=False)) + 0.5
        if not test:
            self.goal_position[:, 0] = GOAL_POSITION_TRAIN[0] # (3.0 * torch.rand(self.num_envs)) + 1.0   
            self.goal_position[:, 1] = GOAL_POSITION_TRAIN[1] 
        else:
            self.goal_position[:, 0] = GOAL_POSITION_VAL[0] # (3.0 * torch.rand(self.num_envs)) + 1.0   
            self.goal_position[:, 1] = GOAL_POSITION_VAL[1]    
        self.extras = {}


        self.reward_scales = {k: v for k, v in vars(reward_scales).items() if not (k.startswith('__') or k.startswith('terminal'))}
        self.terminal_reward_scales = {k: v for k, v in vars(reward_scales).items() if (not k.startswith('__')) and k.startswith('terminal')}
        # print(self.terminal_reward_scales)
        # self.reward_functions = ['distance', 'action_rate']
        
        self._prepare_reward_function()
        
    def _prepare_reward_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        for key in list(self.terminal_reward_scales.keys()):
            scale = self.terminal_reward_scales[key]
            if scale == 0:
                self.terminal_reward_scales.pop(key)
        
        self.reward_functions = []
        self.reward_names = []
        
        self.terminal_reward_functions = []
        self.terminal_reward_names = []

        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        for name, scale in self.terminal_reward_scales.items():
            self.terminal_reward_names.append(name)
            name = '_reward_' + name
            self.terminal_reward_functions.append(getattr(self, name))
        
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in [*self.reward_scales.keys(), *self.terminal_reward_scales.keys()]}
        
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in [*self.reward_scales.keys(), *self.terminal_reward_scales.keys()]}
        
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def distance_control(self):
        return (self.goal_position - self.base_pos)

    def step(self, actions):
        self.actions = torch.clamp(actions, -0.75, 0.75)
        # self.actions[:, :2] *= (torch.norm(self.actions[:, :2], dim=1) > 0.2).unsqueeze(1)
        # self.actions[:, :2] *= (torch.norm(self.actions[:, :2], dim=1) > 0.2).unsqueeze(1)
        # self.actions[:, :2] *= (torch.norm(self.actions[:, :2], dim=1) > 0.2).unsqueeze(1)
        for i in range(STEP_SIZE):
            with torch.no_grad():
                ll_actions = self.low_level_policy(self.ll_obs)

            self.ll_env.commands[:, :3] = self.actions
            # print(self.actions[0], self.ll_env.base_lin_vel[0], self.ll_env.base_ang_vel[0])
            # self.ll_env.commands[:, :3] *= (torch.norm(self.ll_env.commands[:, :3], dim=1) > 0.5).unsqueeze(1)
            self.ll_obs, self.ll_rew, self.ll_dones, self.ll_info = self.ll_env.step(ll_actions)


        # compute observations
        self.episode_length_buf += 1
        # print(self.episode_length_buf[:3])
        # print(self.max_episode_length, self.ll_env.dt)


        self.post_physics_step()
        env_ids = self.check_termination() 
        self.compute_reward()
        self.reset_idx(env_ids)
        self.compute_observations()
        self.last_actions[:] = self.actions[:]

        return { 'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_history': self.obs_history }, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        self.lateral_vel[:] = 0.
        self.backward_vel[:] = 0.
        self.base_pos = self.ll_env.root_states[:, :3] - self.ll_env.env_origins[:, :3] - self.ll_env.base_init_state[:3]
        self.dist_travelled[:] += torch.abs(torch.linalg.norm(self.base_pos[:, :2] - self.last_pos[:, :2], dim=-1))
        self.lateral_vel[:] = self.ll_env.base_lin_vel[:, 1]
        self.backward_vel[:] = torch.clamp_max(self.ll_env.base_lin_vel[:, 0], 0)
        self.angular_vel[:] = self.ll_env.base_ang_vel[:, 2]

    def compute_observations(self):
        self.base_pos = self.ll_env.root_states[:, :3] - self.ll_env.env_origins[:, :3] - self.ll_env.base_init_state[:3]
        self.trajectory[self.episode_length_buf[self.traj_id]-1, :] = self.base_pos[self.traj_id, :2]
        
        self.base_quat = self.ll_env.root_states[:, 3:7]
        self.base_lin_vel = self.ll_env.base_lin_vel.clone()
        self.base_ang_vel = self.ll_env.base_ang_vel.clone()

        self.world_obs = self.ll_env.world_obs

        # self.obs_buf = torch.cat([self.base_pos,  self.base_quat, self.base_lin_vel, self.base_ang_vel, self.actions, self.world_obs], dim=-1)

        if world_cfg.CUSTOM_BLOCK:
            self.obs_buf = torch.cat([self.base_pos, self.base_quat, self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:], self.actions, self.world_obs], dim=-1)
        else:
            self.obs_buf = torch.cat([self.base_pos, self.base_quat, self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:], self.actions], dim=-1)
        
        # print()
        # position_x, position_y, velocity_x, velocity_y, velocity_ang, cmd (vel_x, vel_y, vel_ang), [0] * 9
        
        # rand_num = np.random.choice(torch.arange(0, self.num_envs, 1))
        self.last_pos[:] = self.base_pos[:]
        # print(self.obs_buf[self.traj_id, :2], self.obs_buf[self.traj_id, -4:], self.last_pos[self.traj_id, 0] - (self.goal_position[self.traj_id, 0]))    

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            for i in range(len(self.terminal_reward_functions)):
                name = self.terminal_reward_names[i]
                rew = self.terminal_reward_functions[i]() * self.terminal_reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew

        self.episode_sums["total"] += self.rew_buf
    
    def check_termination(self):
        # base_pos = self.ll_env.root_states[:, :3] - self.ll_env.env_origins[:, :3] - self.ll_env.base_init_state[:3]
        self.gs_buf = torch.linalg.norm(self.base_pos[:, :2] - self.goal_position, dim=-1) < GOAL_THRESHOLD # reach a point goal
        # self.gs_buf = torch.abs(self.base_pos[:, 0] - (self.goal_position[:, 0])) < GOAL_THRESHOLD # break out of region goal
        gs_env_ids = self.gs_buf.nonzero(as_tuple=False).flatten()
        # if len(gs_env_ids) > 100:
        #     print("---------------------------------------------------------------")
        #     print(self.goal_position[gs_env_ids[0]], self.base_pos[gs_env_ids[0], :2], torch.linalg.norm(self.base_pos[gs_env_ids[0], :2] - self.goal_position[gs_env_ids[0], :2], dim=-1), self.episode_length_buf[gs_env_ids[0]], len(gs_env_ids), self.actions[gs_env_ids[0]])
        #     print("---------------------------------------------------------------")


        # print(self.base_pos[0, :2], torch.linalg.norm(self.base_pos[0, :2] - self.goal_position[0, :2], dim=-1), self.episode_length_buf[0], self.actions[0])
        self.time_buf = (self.episode_length_buf >= self.max_episode_length)

        # self.ll_dones |= self.time_buf
        self.reset_buf |= self.ll_dones
        # self.reset_buf |= self.gs_buf
        self.reset_buf |= self.time_buf

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        return env_ids

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return self.obs_buf

        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["train/episode"]['rew_' + key] = torch.mean(self.episode_sums[key][train_env_ids])
                self.episode_sums[key][train_env_ids] = 0
        
        eval_env_ids = env_ids[env_ids >= self.num_train_envs]
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}
            for key in self.episode_sums_eval.keys():
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.extras["eval/episode"]['rew_' + key] = torch.mean(self.episode_sums[key][eval_env_ids])
                self.episode_sums[key][eval_env_ids] = 0

        # if self.traj_id in eval_env_ids:
            # print(self.traj_id)
            # if (self.episode_length_buf[self.traj_id]-1) >= 2:
            #     self.all_trajectories.append(self.trajectory[:self.episode_length_buf[self.traj_id]-1].clone().cpu().numpy())
            #     print('ADDED trajectory', len(self.all_trajectories))
            #     block = self.world_obs[self.traj_id]
            #     print(self.traj_id, block)
            #     block_pos, block_size = block[:2], block[2:]
            #     block_xy = block[:2] - block[2:4]/2
            #     print(block_xy, block_pos) 
            #     # circle = plt.Circle(self.goal_position[self.traj_id].clone().cpu(), GOAL_THRESHOLD, color='blue')
            #     self.all_rects.append(Rectangle(block_xy.cpu(), *block_size.cpu(), color='blue'))
                
            
            # if len(self.all_trajectories) == 7:
                
            #     fig, axes = plt.subplots(3, 2, figsize=(10, 4))
            #     axes = axes.flatten()
            #     for traj, rect, ax in zip(self.all_trajectories[1:], self.all_rects[1:], axes):
            #         ax.set_xlim(-1.0, 4.0)
            #         ax.set_ylim(-1.0, 1.0)
            #         ax.scatter(traj[0, 0], traj[0, 1], color='red')
            #         ax.plot(traj[:, 0], traj[:, 1])
            #         ax.add_patch(rect)
            #         ax.scatter(traj[-1, 0], traj[-1, 1], color='green')
            #         ax.scatter(self.goal_position[self.traj_id, 0].clone().cpu(), self.goal_position[self.traj_id, 1].clone().cpu())
            #     plt.savefig(f'{TRAJ_IMAGE_FOLDER}/traj_{self.traj_image_id}.png')
            #     print(f'plotting now traj_{self.traj_image_id}.png')
            #     plt.close()
            #     self.traj_image_id += 1
            #     self.traj_id = torch.tensor(np.random.uniform(self.num_train_envs, self.num_envs), device=self.device, dtype=torch.int, requires_grad=False)
            #     self.all_trajectories = []
            #     self.all_rects = []
        
                
            # print(self.episode_length_buf[env_ids])
            # self.traj_id = torch.tensor(np.random.uniform(0, self.num_train_envs), device=self.device, dtype=torch.int, requires_grad=False)
            # self.trajectory[:] = 0.

        self.ll_env.reset_idx(env_ids)
        self.rew_buf[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.

        self.lateral_vel[env_ids] = 0.
        self.backward_vel[env_ids] = 0.
        self.dist_travelled[env_ids] = 0. 
        self.last_pos[env_ids] = self.ll_env.root_states[env_ids, :3] - self.ll_env.env_origins[env_ids, :3] - self.ll_env.base_init_state[:3]


        self.compute_observations()
        self.reset_buf[env_ids] = False

        return { 'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_history': self.obs_history }

    def reset(self):
        self.reset_idx(torch.arange(0, self.num_envs, 1, dtype=torch.long, device=self.device))
        return { 'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_history': self.obs_history }

    def reset_evaluation_envs(self):
        env_ids_eval = torch.arange(self.num_train_envs, self.num_envs, 1, dtype=torch.long, device=self.device)
        for key in self.episode_sums_eval.keys():
            unset_eval_envs = env_ids_eval[self.episode_sums_eval[key][env_ids_eval] == -1]
            self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
            ep_sums_key = self.episode_sums_eval[key]
            self.extras["eval/episode"]['rew_' + key] = torch.mean(ep_sums_key[ep_sums_key != -1])
        # for k, v in self.extras['eval/episode'].items():
        #     wandb.log({f'eval/{k}': v})
        self.reset_idx(env_ids_eval)
        for key in self.episode_sums_eval.keys():
            self.episode_sums_eval[key] = -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        return { 'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_history': self.obs_history }

    def _load_env(self, num_envs=1, headless=False):
        # prepare environment
        # config_mini_cheetah(Cfg)
        from ml_logger import logger

        from pathlib import Path
        from mini_gym import MINI_GYM_ROOT_DIR
        import glob
        import os

        recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)
        print(recent_runs)
        logger.configure(Path(recent_runs[-1]).resolve())

        config_go1(Cfg)
        print(logger.glob("*"))
        print(logger.prefix)

        params = logger.load_pkl('parameters.pkl')

        if 'kwargs' in params[0]:
            deps = params[0]['kwargs']

            from mini_gym_learn.ppo.ppo import PPO_Args
            from mini_gym_learn.ppo.actor_critic import AC_Args
            from mini_gym_learn.ppo import RunnerArgs

            AC_Args._update(deps)
            PPO_Args._update(deps)
            RunnerArgs._update(deps)
            Cfg.terrain._update(deps)
            Cfg.commands._update(deps)
            Cfg.normalization._update(deps)
            Cfg.env._update(deps)
            Cfg.domain_rand._update(deps)
            Cfg.rewards._update(deps)
            Cfg.reward_scales._update(deps)
            Cfg.perception._update(deps)
            Cfg.domain_rand._update(deps)
            Cfg.control._update(deps)

        # turn off DR for evaluation script
        Cfg.domain_rand.push_robots = False
        Cfg.domain_rand.randomize_friction = False
        Cfg.domain_rand.randomize_gravity = False
        Cfg.domain_rand.randomize_restitution = False
        Cfg.domain_rand.randomize_motor_offset = False
        Cfg.domain_rand.randomize_motor_strength = False
        Cfg.domain_rand.randomize_friction_indep = False
        Cfg.domain_rand.randomize_ground_friction = False
        Cfg.domain_rand.randomize_base_mass = False
        Cfg.domain_rand.randomize_Kd_factor = False
        Cfg.domain_rand.randomize_Kp_factor = False
        Cfg.domain_rand.randomize_joint_friction = False
        Cfg.domain_rand.randomize_com_displacement = False

        Cfg.env.num_recording_envs = 1
        Cfg.env.num_envs = num_envs
        Cfg.terrain.num_rows = 3
        Cfg.terrain.num_cols = 5
        Cfg.terrain.border_size = 0

        from mini_gym.envs.wrappers.history_wrapper import HistoryWrapper

        env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
        env = HistoryWrapper(env)

        # load policy
        from ml_logger import logger
        from mini_gym_learn.ppo.actor_critic import ActorCritic

        actor_critic = ActorCritic(
            num_obs=Cfg.env.num_observations, 
            num_privileged_obs=Cfg.env.num_privileged_obs,
            num_obs_history=Cfg.env.num_observations * \
                            Cfg.env.num_observation_history,
            num_actions=Cfg.env.num_actions)

        weights = logger.load_torch("checkpoints/ac_weights_002000.pt")
        actor_critic.load_state_dict(state_dict=weights)
        actor_critic.to(env.device)
        policy = actor_critic.act_inference

        return env, policy

    def get_observations(self):
        return { 'obs': self.obs_buf, 'privileged_obs': self.privileged_obs_buf, 'obs_history': self.obs_history }
    
    def _reward_time(self):
        return self.episode_length_buf/self.max_episode_length

    def _reward_distance(self):
        return 1.0 - (1/torch.exp(torch.linalg.norm(self.last_pos[:, :2] - self.goal_position, dim=-1))) # reach a point goal
        # d = torch.abs(self.last_pos[:, 0] - (self.goal_position[:, 0]))
        # r = (d/self.goal_position[:, 0])**0.4 # break out of region goal
        # return r
        # r[d < (GOAL_THRESHOLD*2)] = -2.0
        # print(r[0])
        # print(r[0]    )
        # return torch.linalg.norm(self.last_pos[:, :2] - self.goal_position, dim=-1)

    def _reward_angular_vel(self):
        return torch.abs(self.angular_vel)

    def _reward_lateral_vel(self):
        return torch.square(self.lateral_vel)
    
    def _reward_backward_vel(self):
        return torch.square(self.backward_vel)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    # def _reward_action_rate(self):
    #     return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_ll_reset(self):
        return self.ll_dones * 1.0

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.ll_env.contact_forces[:, self.ll_env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)
    
    def _reward_distance_gs(self):
        # print(self.gs_buf * 1.0)
        return self.gs_buf * 1.0

    def _reward_velocity_gs(self):
        # print(self.gs_buf * 1.0)
        return self.gs_buf * (1.0/torch.exp(torch.linalg.norm(self.base_lin_vel, dim=-1) + torch.linalg.norm(self.base_ang_vel, dim=-1)))

    
    def _reward_terminal_ll_reset(self):
        return self.ll_dones * 1.0

    def _reward_terminal_distance_gs(self):
        # print(self.gs_buf * 1.0)
        return self.gs_buf * self.reset_buf * 1.0

    def _reward_terminal_gs_velocity(self):
        # print(self.gs_buf * 1.0)
        return (self.time_buf & self.gs_buf) * (1.0 - 1/torch.exp(torch.linalg.norm(self.base_lin_vel, dim=-1) + torch.linalg.norm(self.base_ang_vel, dim=-1)))
    
    def _reward_terminal_distance_covered(self):
        # print('terminal')
        # print(self.dist_travelled * 1.0)
        # return self.gs_buf * (self.dist_travelled/torch.norm(self.ll_env.go1_init_states - self.goal_position, dim=-1))
        return self.gs_buf * self.dist_travelled

    def _reward_terminal_time_out(self):
        return (self.time_buf & ~self.gs_buf) * 1.0

    