# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import torch

from mini_gym_learn.utils import split_and_pad_trajectories
from high_level_policy import SAVE_ADAPTATION_DATA, ROLLOUT_HISTORY, SAVE_ADAPTATION_DATA_FILE_NAME
from ml_logger import logger
from pathlib import Path
from datetime import datetime
import pickle
import time
import numpy as np
import threading
import queue

# thread that consumes the data from the queue to save to disk
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
                if isinstance(v, torch.Tensor):
                    item[k] = v.cpu().numpy()
            np.savez_compressed(self.data_path/f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz', **item)
            
            # with open(SAVE_ADAPTATION_DATA_FILE_NAME, 'wb') as f:
            #     pickle.dump(item, f)

            # send a signal to the queue that the job is done
            self.queue.task_done()


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.observation_histories = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.env_bins = None
            self.adaptation_hidden_states = None
            self.actor_hidden_states = None
            self.critic_hidden_states = None
            self.latent_teacher_states = None
            self.full_seen_world = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, obs_history_shape, actions_shape, adaptation_hidden_sizes, latent_size, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.full_seen_world_size = privileged_obs_shape
        self.obs_history_shape = obs_history_shape
        self.actions_shape = actions_shape
        self.adaptation_hidden_sizes = adaptation_hidden_sizes
        self.latent_size = latent_size
        self.actor_hidden_size = [256]
        self.critic_hidden_size = [256]

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        self.full_seen_world = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        self.observation_histories = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device)
        self.adaptation_hidden_states = torch.zeros(num_transitions_per_env, num_envs, *self.adaptation_hidden_sizes, device=self.device)
        
        self.actor_hidden_states = torch.zeros(1, num_envs, *self.actor_hidden_size, device=self.device)
        self.critic_hidden_states = torch.zeros(1, num_envs, *self.critic_hidden_size, device=self.device)

        self.latent_teacher_states = torch.zeros(num_transitions_per_env, num_envs, *self.latent_size, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.env_bins = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs


        if SAVE_ADAPTATION_DATA:
            self.q = queue.Queue(maxsize=200)
            self.worker = Worker(self.q, fn=print)
            self.worker.start()
            self.q.join()

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.full_seen_world[self.step].copy_(transition.full_seen_world)
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.adaptation_hidden_states[self.step].copy_(transition.adaptation_hidden_states)
        self.latent_teacher_states[self.step].copy_(transition.latent_teacher_states)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        # self.env_bins[self.step].copy_(transition.env_bins.view(-1, 1))
        self.step += 1

    def clear(self, save=False):
        # saves the observations, privileged_observations, observation_histories, actions, dones to a file using pickle
        if save and SAVE_ADAPTATION_DATA:
            # create a dictionary of the data to save
            # print('saving data')
            # start = time.time()
            data = {
                'observations': self.observations.clone(),
                'privileged_observations': self.privileged_observations.clone(),
                'observation_histories': self.observation_histories[:, :, -(self.observation_histories.shape[2]//ROLLOUT_HISTORY):].clone(),
                'full_seen_world': self.full_seen_world.clone(),
                'dones': self.dones.clone()
            }

            self.q.put(data)
            
            # data_path = Path(f'/common/users/dm1487/legged_manipulation_data/rollout_data/{SAVE_ADAPTATION_DATA_FILE_NAME}')

            # make directory `data_path` if it doesn't exist
            # data_path.mkdir(parents=True, exist_ok=True)

            # np.savez_compressed(data_path/f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz', **data)

            # print(time.time()-start)

            # # save data to file using timestamp as name in `data_path` folder using pickle
            # with open(data_path / f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl', 'wb') as f:
            #     pickle.dump(data, f)
        self.step = 0
        

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        # print(mini_batch_size)
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)
        observations = self.observations.flatten(0, 1)
        privileged_obs = self.privileged_observations.flatten(0, 1)
        obs_history = self.observation_histories.flatten(0, 1)
        adaptation_hidden_states = self.adaptation_hidden_states.flatten(0, 1)
        latent_teacher_states = self.latent_teacher_states.flatten(0, 1)
        critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        # old_env_bins = self.env_bins.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                privileged_obs_batch = privileged_obs[batch_idx]
                obs_history_batch = obs_history[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                adaptation_hidden_states_batch = adaptation_hidden_states[batch_idx]
                latent_teacher_states_batch = latent_teacher_states[batch_idx]
                # env_bins_batch = old_env_bins[batch_idx]
                yield obs_batch, critic_observations_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, adaptation_hidden_states_batch, None, None # env_bins_batch

    # for RNNs only
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        padded_privileged_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.privileged_observations, self.dones)
        padded_obs_history_trajectories, trajectory_masks = split_and_pad_trajectories(self.observation_histories, self.dones)
        padded_critic_obs_trajectories = padded_obs_trajectories

        # print(padded_obs_trajectories.shape, padded_privileged_obs_trajectories.shape, padded_obs_history_trajectories.shape, padded_critic_obs_trajectories.shape)

        mini_batch_size = self.num_envs // num_mini_batches

        # print(mini_batch_size)
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]
                obs_history_batch = padded_obs_history_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                yield obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, masks_batch, None, None
                
                first_traj = last_traj