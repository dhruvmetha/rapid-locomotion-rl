# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from high_level_policy.ppo import ActorCritic
from high_level_policy.ppo import RolloutStorage
from high_level_policy.ppo import caches

from high_level_policy import USE_LATENT, DECODER

torch.manual_seed(42)

class PPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.9999
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPO_Args.learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                      lr=PPO_Args.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPO_Args.learning_rate
        self.iters = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history, student=False):
        # Compute the actions and values
        priv_latent, actions = self.actor_critic.act(obs, privileged_obs, student=student, observation_history=obs_history)
        self.transition.actions = actions.detach()
        self.transition.values = self.actor_critic.evaluate(obs, privileged_obs, student=student, observation_history=obs_history).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        # print('adding to transition', obs_history[0, -60:])
        self.transition.observation_histories = obs_history
        return priv_latent, self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        # since it timed-out, it does not get future true rewards anymore for whatever trajectory it may have taken
        # thus we bootstrap the (maybe true) future reward using the value function. (induces bias)
        if 'time_outs' in infos:
            self.transition.rewards += PPO_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        # print('saving to rollout store', self.transition.observation_histories[0, -60:])
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs, **kwargs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs, **kwargs).detach()
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)


    def decoder_loss(self, gt, pred):
        
        # reconstruction_loss = torch.tensor(0).float().to(self.device)

        contact_gt = gt[:, 0::10]
        movable_gt = gt[:, 1::10]

        contact_pred = F.sigmoid(pred[:, 0::10])
        movable_pred = F.sigmoid(pred[:, 1::10])

        local_loss = lambda inp, tar : F.binary_cross_entropy(inp, tar)

        # reconstruction_loss += local_loss(contact_pred, contact_gt) + local_loss(movable_pred, movable_gt)
        
        extract_idx = [torch.arange(0, 10, dtype=torch.long, device=self.device), torch.arange(10, 20, dtype=torch.long, device=self.device), torch.arange(20, 30, dtype=torch.long, device=self.device), torch.arange(30, 40, dtype=torch.long, device=self.device)]
        reconstruction_loss = None
        for i in extract_idx:
            loss = F.binary_cross_entropy(F.sigmoid(pred[:, i[0]]), gt[:, i[0]]) 
            loss += F.binary_cross_entropy(F.sigmoid(pred[:, i[1]]), gt[:, i[1]])
            loss += F.mse_loss(pred[:, i[2:]], gt[:, i[2:]])

            if reconstruction_loss is not None:
                reconstruction_loss += loss
            else:
                reconstruction_loss = loss

        # fixed_rects_full_idx = torch.cat([torch.arange(2, 10, dtype=torch.long, device=self.device), torch.arange(12, 20, dtype=torch.long, device=self.device), torch.arange(22, 30, dtype=torch.long, device=self.device)])
        # bs, _ = gt.shape
        # movable_rects_idx = torch.arange(0, 10, dtype=torch.long, device=self.device).repeat(bs, 1)
        # movable_rects = torch.zeros(bs, 10, device=self.device)
        # fixed_rects_idx = torch.arange(0, 30, dtype=torch.long, device=self.device).repeat(bs, 1)
        # fixed_rects = torch.zeros(bs, 30, device=self.device)



        # contact_mask = gt[:, 0::10] == 1
        # movable_mask = gt[:, 1::10] == 1

        # movable_contact = contact_mask * movable_mask
        # ba, ba_idx = movable_contact.nonzero(as_tuple=True)
        # if ba.size(0) > 0:
        #     print('here1')
        #     movable_rects_idx = movable_rects_idx[ba, :] + (ba_idx * 10).view(-1, 1)
        #     print('here2')
        #     movable_rects[ba, :] = gt[ba.view(-1, 1), movable_rects_idx]

        # # split_gt = gt.view(-1, 4, 10)

        # # for i in range(3):

        # fixed_rects_idx = torch.arange(0, 30, dtype=torch.long, device=self.device).repeat(bs, 1)
        # fixed_contact = contact_mask * (~movable_mask)
        # ba, ba_idx = fixed_contact.nonzero(as_tuple=True)
        # if ba.size(0) > 0:
        #     print('here3', fixed_rects_idx[ba, :], ba, ba_idx)
        #     fixed_rects_idx = (fixed_rects_idx[ba, :] + ((ba_idx) * 10).view(-1, 1))[:, :10]
        #     print('here4', fixed_rects_idx)
        #     fixed_rects[ba.view(-1, 1), fixed_rects_idx] = gt[ba.view(-1, 1), fixed_rects_idx]
        #     print('here5')

        # movable_rect_pred = pred[:, :10]
        # fixed_rect_pred =  pred[:, 10:]

        # contact_movable_rect_gt = movable_rects[:, 0]
        # moving_movable_rect_gt = movable_rects[:, 1]

        # contact_fixed_rect_gt = fixed_rects[:, 0::10]
        # moving_fixed_rect_gt = fixed_rects[:, 1::10]
        
        # contact_movable_pred = F.sigmoid(movable_rect_pred[:, 0])
        # moving_movable_pred = F.sigmoid(movable_rect_pred[:, 1])
        
        # contact_fixed_pred = F.sigmoid(fixed_rect_pred[:, 0::10])
        # moving_fixed_pred = F.sigmoid(fixed_rect_pred[:, 1::10])

        # local_loss = lambda inp, tar : F.binary_cross_entropy(inp, tar)

        # reconstruction_loss = local_loss(contact_movable_pred, contact_movable_rect_gt) + local_loss(moving_movable_pred, moving_movable_rect_gt) + local_loss(contact_fixed_pred, contact_fixed_rect_gt) + local_loss(moving_fixed_pred, moving_fixed_rect_gt)

        # reconstruction_loss += F.mse_loss(movable_rect_pred[:, 2:], movable_rects[:, 2:]) + F.mse_loss(fixed_rect_pred[:, fixed_rects_full_idx], fixed_rects[:, fixed_rects_full_idx])

        return reconstruction_loss

    def update(self, student=False):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_reconstruction_loss = 0
        mean_adaptation_module_loss = 0
        mean_adaptation_reconstruction_loss = 0
        old_adaptation_target = None

        mean_entropy_loss = 0
        mean_kl = 0

        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            ((priv_train_pred_teacher, priv_train_pred_student), (latent_enc_teacher, latent_enc_student)), _ = self.actor_critic.act(obs_batch, privileged_obs_batch, student=student, observation_history=obs_history_batch, masks=masks_batch)
            
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, privileged_obs_batch, student=student, observation_history=obs_history_batch, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if PPO_Args.desired_kl != None and PPO_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    mean_kl += kl_mean.item()

                    if kl_mean > PPO_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPO_Args.clip_param,
                                                                               1.0 + PPO_Args.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            
            loss = surrogate_loss + PPO_Args.value_loss_coef * value_loss - PPO_Args.entropy_coef * entropy_batch.mean()

            with torch.inference_mode():
                mean_entropy_loss += PPO_Args.entropy_coef * entropy_batch.mean().item()

            if USE_LATENT:
                if DECODER:
                    
                    # fixed_rects_full_idx = torch.cat([torch.arange(2, 10, dtype=torch.long, device=self.device), torch.arange(12, 20, dtype=torch.long, device=self.device), torch.arange(22, 30, dtype=torch.long, device=self.device)])

                    # bs, _ = privileged_obs_batch.shape

                    # movable_rects_idx = torch.arange(0, 10, dtype=torch.long, device=self.device).repeat(bs, 1)
                    # movable_rects = torch.zeros(bs, 10, device=self.device)
                    # fixed_rects_idx = torch.arange(0, 30, dtype=torch.long, device=self.device).repeat(bs, 1)
                    # fixed_rects = torch.zeros(bs, 30, device=self.device)
                    # contact_mask = privileged_obs_batch[:, 0::10] == 1
                    # movable_mask = privileged_obs_batch[:, 1::10] == 1
                    # movable_contact = contact_mask * movable_mask
                    # ba, ba_idx = movable_contact.nonzero(as_tuple=True)
                    
                    # if ba.size(0) > 0:
                    #     movable_rects_idx = movable_rects_idx[ba, :] + (ba_idx * 10).view(-1, 1)
                    #     movable_rects[ba, :] = privileged_obs_batch[ba.view(-1, 1), movable_rects_idx]

                    # fixed_contact = contact_mask * (~movable_mask)
                    # ba, ba_idx = fixed_contact.nonzero(as_tuple=True)
                    # if ba.size(0) > 0:
                    #     fixed_rects_idx = fixed_rects_idx[ba, :] + (ba_idx * 10).view(-1, 1)
                    #     fixed_rects[ba, :] = privileged_obs_batch[ba.view(-1, 1), fixed_rects_idx]


                    # movable_rect_pred = priv_train_pred_teacher[:, :10]
                    # fixed_rect_pred =  priv_train_pred_teacher[:, 10:]

                    # contact_movable_rect_gt = movable_rects[:, 0]
                    # moving_movable_rect_gt = movable_rects[:, 1]

                    # contact_fixed_rect_gt = fixed_rects[:, 0]
                    # moving_fixed_rect_gt = fixed_rects[:, 1]
                    
                    # contact_movable_pred = torch.nn.Sigmoid(movable_rect_pred[:, 0])
                    # moving_movable_pred = torch.nn.Sigmoid(movable_rect_pred[:, 1])
                    
                    # contact_fixed_pred = torch.nn.Sigmoid(fixed_rect_pred[:, 0::10])
                    # moving_fixed_pred = torch.nn.Sigmoid(fixed_rect_pred[:, 1::10])

                    # local_loss = lambda inp, tar : F.binary_cross_entropy(inp, tar)


                    # reconstruction_loss = local_loss(contact_movable_pred, contact_movable_rect_gt) + local_loss(moving_movable_pred, moving_movable_rect_gt) + local_loss(contact_fixed_pred, contact_fixed_rect_gt) + local_loss(moving_fixed_pred, moving_fixed_rect_gt)

                    # reconstruction_loss += F.mse_loss(movable_rect_pred[:, 2:], movable_rects[:, 2:]) + F.mse_loss(fixed_rect_pred[:, fixed_rects_full_idx], fixed_rects[:, fixed_rects_full_idx])

                    reconstruction_loss = self.decoder_loss(privileged_obs_batch, priv_train_pred_teacher)
                    # print(reconstruction_loss.item())

                    # reconstruction_loss = F.mse_loss(privileged_obs_batch, priv_train_pred_teacher)
                    loss += reconstruction_loss
                    # mean_reconstruction_loss += reconstruction_loss.item()

                if student:
                    with torch.no_grad():
                        adaptation_target = self.actor_critic.env_factor_encoder(privileged_obs_batch)
                    # adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
                    adaptation_loss = F.mse_loss(latent_enc_student, adaptation_target)
                    loss += adaptation_loss
                    
                    if DECODER:
                        with torch.no_grad():
                            adaptation_decoded = self.actor_critic.env_factor_decoder(latent_enc_student)
                            adaptation_reconstruction_loss = self.decoder_loss(privileged_obs_batch, adaptation_decoded)
                    # loss += adaptation_reconstruction_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_Args.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            if USE_LATENT:
                if DECODER:
                    mean_reconstruction_loss += reconstruction_loss.item()
                    if student:
                        mean_adaptation_reconstruction_loss += adaptation_reconstruction_loss.item()
                if student:        
                    mean_adaptation_module_loss += adaptation_loss.item()

                if not student:
                    # Adaptation module gradient step
                    for epoch in range(PPO_Args.num_adaptation_module_substeps):
                        # with torch.enable_grad():
                        # adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
                        adaptation_pred = self.actor_critic.get_latent_student(obs_history_batch)
                        
                        with torch.no_grad():
                            adaptation_target = self.actor_critic.env_factor_encoder(privileged_obs_batch)
                            if DECODER:
                                adaptation_decoded = self.actor_critic.env_factor_decoder(adaptation_pred)
                                adaptation_reconstruction_loss = self.decoder_loss(privileged_obs_batch, adaptation_decoded)

                        adaptation_loss = F.mse_loss(adaptation_pred, adaptation_target)
                        total_adap_loss = adaptation_loss
                        
                        # if DECODER:
                        #     total_adap_loss += (adaptation_reconstruction_loss)

                        self.adaptation_module_optimizer.zero_grad()
                        total_adap_loss.backward()
                        self.adaptation_module_optimizer.step()

                        mean_adaptation_module_loss += adaptation_loss.item()
                        if DECODER:
                            mean_adaptation_reconstruction_loss += adaptation_reconstruction_loss.item()
                        
                # else:
                #     for epoch in range(PPO_Args.num_adaptation_module_substeps):
                #         adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
                #         with torch.no_grad():
                #             adaptation_target = self.actor_critic.env_factor_encoder(privileged_obs_batch)
                #             # residual = (adaptation_target - adaptation_pred).norm(dim=1)
                #             # caches.slot_cache.log(env_bins_batch[:, 0].cpu().numpy().astype(np.uint8),
                #             #                       sysid_residual=residual.cpu().numpy())

                #         adaptation_loss = F.mse_loss(adaptation_pred, adaptation_target)

                #         self.adaptation_module_optimizer.zero_grad()
                #         adaptation_loss.backward()
                #         self.adaptation_module_optimizer.step()

                #         mean_adaptation_module_loss += adaptation_loss.item()
                    

        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_reconstruction_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates

        if USE_LATENT:
            if not student:
                mean_adaptation_module_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
                mean_adaptation_reconstruction_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
            else:
                mean_adaptation_module_loss /= num_updates
                mean_adaptation_reconstruction_loss /= num_updates
        self.storage.clear()

        self.iters += 1

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_kl, mean_adaptation_module_loss, mean_reconstruction_loss, mean_adaptation_reconstruction_loss


