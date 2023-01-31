# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader


from high_level_policy import *
from high_level_policy.utils import unpad_trajectories

# np.random.seed(42)
torch.manual_seed(42)

class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    # actor_hidden_dims = [2048, 2048, 1024, 512, 256, 128]
    # critic_hidden_dims = [2048, 2048, 1024, 512, 256, 128]
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    shared_hidden_dims = [128, 13]
    activation = 'tanh'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [[4096, 2048, 1024, 512, 256, 128, 64]]

    env_factor_encoder_branch_input_dims = [PER_RECT*2 if world_cfg.fixed_block.add_to_obs else 9]
    env_factor_encoder_branch_latent_dims = [LATENT_DIM_SIZE if world_cfg.fixed_block.add_to_obs else 4]
    env_factor_encoder_branch_hidden_dims = [[256, 128]]

    env_factor_decoder_branch_input_dims = [LATENT_DIM_SIZE if world_cfg.fixed_block.add_to_obs else 4]
    env_factor_decoder_branch_latent_dims = [PER_RECT*2 if world_cfg.fixed_block.add_to_obs else 9]
    env_factor_decoder_branch_hidden_dims = [[256, 128]]


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super().__init__()

        activation = get_activation(AC_Args.activation)

        self.num_privileged_obs = num_privileged_obs
        self.num_obs_history = num_obs_history


        self.hidden_states = None
        
        total_latent_dim = 0
        if USE_LATENT:
            if OCCUPANCY_GRID:
                for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                        zip(AC_Args.env_factor_encoder_branch_input_dims,
                            AC_Args.env_factor_encoder_branch_hidden_dims,
                            AC_Args.env_factor_encoder_branch_latent_dims)):
                    # Env factor encoder
                    env_factor_encoder_layers = []

                    env_factor_encoder_layers.append(
                        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1))

                    
                    env_factor_encoder_layers.append(activation)

                    
                    for l in range(len(branch_hidden_dims)):
                        if l == len(branch_hidden_dims) - 1:
                            
                            env_factor_encoder_layers.append(
                                nn.Conv2d(in_channels=branch_hidden_dims[l],
                                        out_channels=branch_latent_dim,
                                        kernel_size=3,
                                        stride=1))
                        else:
                            
                            env_factor_encoder_layers.append(
                                nn.Conv2d(in_channels=branch_hidden_dims[l],
                                        out_channels=branch_hidden_dims[l + 1],
                                        kernel_size=3,
                                        stride=1))

                            env_factor_encoder_layers.append(activation)

                self.env_factor_encoder = nn.Sequential(*env_factor_encoder_layers)
                self.add_module(f"encoder", self.env_factor_encoder)


                for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                        zip(AC_Args.env_factor_decoder_branch_input_dims,
                            AC_Args.env_factor_decoder_branch_hidden_dims,
                            AC_Args.env_factor_decoder_branch_latent_dims)):
                    # Env factor decoder
                    env_factor_decoder_layers = []

                   
                    env_factor_decoder_layers.append(
                        nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=3, stride=1))

                    env_factor_decoder_layers.append(activation)

                    
                    for l in range(len(branch_hidden_dims)):
                        if l == len(branch_hidden_dims) - 1:
                            
                            env_factor_decoder_layers.append(
                                nn.ConvTranspose2d(in_channels=branch_hidden_dims[l],
                                                out_channels=branch_latent_dim,
                                                kernel_size=3,
                                                stride=1))
                        else:
                            env_factor_decoder_layers.append(
                                nn.ConvTranspose2d(in_channels=branch_hidden_dims[l],
                                                out_channels=branch_hidden_dims[l + 1],
                                                kernel_size=3,
                                                stride=1))
                            
                            env_factor_decoder_layers.append(activation)
                self.env_factor_decoder = nn.Sequential(*env_factor_decoder_layers)
                self.add_module(f"decoder", self.env_factor_decoder)

            else:

                if ENCODER:
                    for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                            zip(AC_Args.env_factor_encoder_branch_input_dims,
                                AC_Args.env_factor_encoder_branch_hidden_dims,
                                AC_Args.env_factor_encoder_branch_latent_dims)):
                        # Env factor encoder
                        env_factor_encoder_layers = []
                        env_factor_encoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
                        env_factor_encoder_layers.append(activation)
                        for l in range(len(branch_hidden_dims)):
                            if l == len(branch_hidden_dims) - 1:
                                env_factor_encoder_layers.append(
                                    nn.Linear(branch_hidden_dims[l], branch_latent_dim))
                            else:
                                env_factor_encoder_layers.append(
                                    nn.Linear(branch_hidden_dims[l],
                                            branch_hidden_dims[l + 1]))
                                env_factor_encoder_layers.append(activation)
                    self.env_factor_encoder = nn.Sequential(*env_factor_encoder_layers)
                    self.add_module(f"encoder", self.env_factor_encoder)

                    if DECODER:
                        for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                                zip(AC_Args.env_factor_decoder_branch_input_dims,
                                    AC_Args.env_factor_decoder_branch_hidden_dims,
                                    AC_Args.env_factor_decoder_branch_latent_dims)):
                            # Env factor decoder
                            env_factor_decoder_layers = []
                            env_factor_decoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
                            env_factor_decoder_layers.append(activation)
                            for l in range(len(branch_hidden_dims)):
                                if l == len(branch_hidden_dims) - 1:
                                    env_factor_decoder_layers.append(
                                        nn.Linear(branch_hidden_dims[l], branch_latent_dim))
                                else:
                                    env_factor_decoder_layers.append(
                                        nn.Linear(branch_hidden_dims[l],
                                                branch_hidden_dims[l + 1]))
                                    env_factor_decoder_layers.append(activation)
                        self.env_factor_decoder = nn.Sequential(*env_factor_decoder_layers)
                        self.add_module(f"decoder", self.env_factor_decoder)

            if ENCODER:
                output_size = LATENT_DIM_SIZE
            else:
                output_size = self.num_privileged_obs

            if not LSTM_ADAPTATION:
                # Adaptation module
                for i, (branch_hidden_dims, branch_latent_dim) in enumerate(zip(AC_Args.adaptation_module_branch_hidden_dims,
                                                                                AC_Args.env_factor_encoder_branch_latent_dims)):
                    adaptation_module_layers = []
                    adaptation_module_layers.append(nn.Linear(num_obs_history, branch_hidden_dims[0]))
                    adaptation_module_layers.append(activation)
                    for l in range(len(branch_hidden_dims)):
                        if l == len(branch_hidden_dims) - 1:
                            adaptation_module_layers.append(
                                nn.Linear(branch_hidden_dims[l], branch_latent_dim))
                        else:
                            adaptation_module_layers.append(
                                nn.Linear(branch_hidden_dims[l],
                                        branch_hidden_dims[l + 1]))
                            adaptation_module_layers.append(activation)
                self.adaptation_module = nn.Sequential(*adaptation_module_layers)

            else:
                # Adaptation module using RNN
                class GRU(nn.Module):
                    def __init__(self, input_size, hidden_size, output_size):
                        super(GRU, self).__init__()
                        self.hidden_size = hidden_size
                        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
                        fc_layers = [hidden_size, output_size]
                        layers_list = []
                        for l in range(len(fc_layers) - 1):
                            layers_list.append(nn.Linear(fc_layers[l], fc_layers[l + 1]))
                            if l < len(fc_layers) - 2:
                                layers_list.append(activation)
                        self.fc = nn.Sequential(*layers_list)

                    def forward(self, observation_history, hidden_states):
                        # print(observation_history.shape)
                        out, h = self.gru(observation_history, hidden_states)
                        # print(h.shape)
                        # hidden_states = h[:, -1, :]
                        # print(out.shape)
                        final_out = self.fc(out[:, -1, :])
                        return final_out, out[:, 1, :]

                # if ENCODER:
                #     output_size = LATENT_DIM_SIZE
                # else:
                #     output_size = self.num_privileged_obs
                
                self.adaptation_module = GRU(num_obs_history//ROLLOUT_HISTORY, HIDDEN_STATE_SIZE, output_size)
                # self.adaptation_module = nn.Identity()
                self.add_module(f"adaptation_module", self.adaptation_module)

            total_latent_dim += output_size

            # Shared layers
            if SHARED:
                shared_layers = []
                shared_layers.append(nn.Linear(total_latent_dim + num_obs, AC_Args.shared_hidden_dims[0]))
                shared_layers.append(activation)
                for l in range(len(AC_Args.shared_hidden_dims)):
                    if l == len(AC_Args.shared_hidden_dims) - 1:
                        shared_layers.append(nn.Linear(AC_Args.shared_hidden_dims[l], AC_Args.shared_hidden_dims[l]))
                    else:
                        shared_layers.append(nn.Linear(AC_Args.shared_hidden_dims[l], AC_Args.shared_hidden_dims[l + 1]))
                        shared_layers.append(activation)
                total_latent_dim = 0

        if ENCODER:
            num_obs_complete = num_obs + total_latent_dim
        else:
            if USE_LATENT:
                num_obs_complete = num_obs + num_privileged_obs
            else:
                num_obs_complete = num_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs_complete, AC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_obs_complete, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)
        
        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        # self.adaptation_module.reset(dones)
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_latent_student(self, observation_history, hidden_states=None):
        if hidden_states is None:
            hidden_states = torch.zeros(1, observation_history.shape[0], HIDDEN_STATE_SIZE).to(observation_history.device)
                
        if not LSTM_ADAPTATION:
            return self.adaptation_module(observation_history), hidden_states.unsqueeze(0)
        else:
            # print(hidden_states.shape)
            observation_history_ = observation_history.view(-1, ROLLOUT_HISTORY, self.num_obs_history//ROLLOUT_HISTORY)
            return self.adaptation_module(observation_history_, hidden_states.unsqueeze(0))

    def update_distribution(self, observations, privileged_observations, rollout=False, student=False, observation_history=None, adaptation_hs=None):
        full_priv_obs_pred_student, full_priv_obs_pred_teacher = None, privileged_observations
        full_latent_teacher, full_latent_student = None, None
        full_next_hidden_states = None
        full_mean = None

        # print(adaptation_hs.shape)

        tensor_dataset = TensorDataset(observations, privileged_observations)
        if observation_history is not None:
            tensor_dataset = TensorDataset(observations, privileged_observations, observation_history, adaptation_hs)
        
        bs = observations.shape[0]
        if rollout:
            # print("Rollout", bs)
            bs = observations.shape[0]
        last_batch_idx = len(tensor_dataset)//bs
        left_over = len(tensor_dataset) - last_batch_idx * bs
            # print(last_batch_idx, left_over)
        # use a batch to do inference if rollout is true
        dataloader = DataLoader(tensor_dataset, batch_size=bs, shuffle=False)

        for batch_idx, data in enumerate(dataloader):


            if observation_history is not None:
                # print(batch_idx)
                if batch_idx == last_batch_idx:
                    observations_batch, privileged_observations_batch, observation_history_batch, adaptation_hs_batch = data[:left_over]
                    # print(observations_batch.size())
                else:
                    observations_batch, privileged_observations_batch, observation_history_batch, adaptation_hs_batch = data
            else:
                observations_batch, privileged_observations_batch = data
            
            if USE_LATENT:
                if ENCODER:
                    latent_teacher = self.env_factor_encoder(privileged_observations_batch)
                else:
                    latent_teacher = privileged_observations_batch
                if ENCODER and DECODER:
                    priv_obs_pred_teacher = self.env_factor_decoder(latent_teacher)
                
                if student:
                    latent_student, next_hidden_states = self.get_latent_student(observation_history_batch, adaptation_hs_batch)
                    mean = self.actor_body(torch.cat((observations_batch, latent_student), dim=-1))
                    # if DECODER:
                    #     priv_obs_pred_student = self.env_factor_decoder(latent_student)
                else:
                    with torch.no_grad():
                        latent_student, next_hidden_states = self.get_latent_student(observation_history_batch, adaptation_hs_batch)
                    # print('next_hidden_states', next_hidden_states.shape)
                    mean = self.actor_body(torch.cat((observations_batch, latent_teacher), dim=-1))

                if full_latent_student is None and full_latent_teacher is None:
                    full_latent_student = latent_student
                    full_latent_teacher = latent_teacher
                    # full_priv_obs_pred_student = priv_obs_pred_student
                    if ENCODER and DECODER:
                        full_priv_obs_pred_teacher = priv_obs_pred_teacher
                    full_mean = mean
                    full_next_hidden_states = next_hidden_states
                    # print('full_next_hidden_states', full_next_hidden_states.shape)
                else:
                    full_latent_student = torch.cat((full_latent_student, latent_student), dim=0)
                    full_latent_teacher = torch.cat((full_latent_teacher, latent_teacher), dim=0)
                    # full_priv_obs_pred_student = torch.cat((full_priv_obs_pred_student, priv_obs_pred_student), dim=0)
                    if ENCODER and DECODER:
                        full_priv_obs_pred_teacher = torch.cat((full_priv_obs_pred_teacher, priv_obs_pred_teacher), dim=0)
                    full_mean = torch.cat((full_mean, mean), dim=0)
                    # print('next_hidden_states', next_hidden_states.shape)
                    full_next_hidden_states = torch.cat((full_next_hidden_states, next_hidden_states), dim=0)
                
            else:
                mean = self.actor_body(observations_batch)
                next_hidden_states = torch.zeros(1, observation_history_batch.shape[0], HIDDEN_STATE_SIZE).to(observation_history_batch.device)

                if full_mean is None:
                    full_mean = mean
                    full_next_hidden_states = next_hidden_states
                else:
                    full_mean = torch.cat((full_mean, mean), dim=0)
                    full_next_hidden_states = torch.cat((full_next_hidden_states, next_hidden_states), dim=0)

            # print('hidden', full_next_hidden_states.shape)
            

        # if USE_LATENT:
        #     # if student:
        #     #     if observation_history is None:
        #     #         raise "No observation history"
        #         # priv_obs_pred_student = self.env_factor_decoder(latent_student)
        #     if ENCODER:
        #         latent_teacher = self.env_factor_encoder(privileged_observations)
        #     else:
        #         latent_teacher = privileged_observations
        #     if ENCODER and DECODER:
        #         priv_obs_pred_teacher = self.env_factor_decoder(latent_teacher)
            
        #     if student:
        #         latent_student, next_hidden_states = self.get_latent_student(observation_history, adaptation_hs)
        #         mean = self.actor_body(torch.cat((observations, latent_student), dim=-1))
        #         # if DECODER:
        #         #     priv_obs_pred_student = self.env_factor_decoder(latent_student)
        #     else:
        #         with torch.no_grad():
        #             latent_student, next_hidden_states = self.get_latent_student(observation_history, adaptation_hs)
                    
        #         mean = self.actor_body(torch.cat((observations, latent_teacher), dim=-1))
                
        # else:
        #     mean = self.actor_body(observations)
        #     next_hidden_states = torch.zeros(1, observation_history.shape[0], HIDDEN_STATE_SIZE).to(observation_history.device)
        self.distribution = Normal(full_mean, full_mean * 0. + self.std)
        return (full_priv_obs_pred_teacher, full_priv_obs_pred_student), (full_latent_teacher, full_latent_student), (full_next_hidden_states.squeeze(0))

    def act(self, observations, privileged_observations, rollout=False, **kwargs):
        priv_obs_pred, latent, hidden_states = self.update_distribution(observations, privileged_observations, rollout=rollout, student=kwargs['student'], observation_history=kwargs['observation_history'], adaptation_hs=kwargs['adaptation_hs'])
        return (priv_obs_pred, latent, hidden_states), self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs"], ob["privileged_obs"])[1]

    def act_inference(self, ob, policy_info={}):
        if USE_LATENT:
            if ob["privileged_obs"] is not None:
                gt_latent = self.env_factor_encoder(ob["privileged_obs"])
                policy_info["gt_latents"] = gt_latent.detach().cpu().numpy()
        return self.act_student(ob["obs"], ob["obs_history"])

    def act_inference_expert(self, ob, policy_info={}):
        if USE_LATENT:
            if ob["privileged_obs"] is not None:
                gt_latent = self.env_factor_encoder(ob["privileged_obs"])
                policy_info["gt_latents"] = gt_latent.detach().cpu().numpy()
        return self.act_teacher(ob["obs"], ob["privileged_obs"])

    def act_student(self, observations, observation_history, adaptation_hs=None, policy_info={}):
        priv_obs_pred = None
        latent = None
        if USE_LATENT:
            # print('here', observation_history.shape)
            # latent = self.adaptation_module(observation_history)
            latent, next_hidden_states = self.get_latent_student(observation_history, adaptation_hs)
            priv_obs_pred = None
            if ENCODER: 
                if DECODER:
                    priv_obs_pred = self.env_factor_decoder(latent)
            else:
                priv_obs_pred = latent

            actions_mean = self.actor_body(torch.cat((observations, latent), dim=-1))
            policy_info["latents"] = latent.cpu().numpy()
        else:
            actions_mean = self.actor_body(observations)
        return (priv_obs_pred, latent, next_hidden_states), actions_mean

    def act_teacher(self, observations, privileged_info, policy_info={}):
        priv_obs_pred = None
        latent = None
        if USE_LATENT:
            priv_obs_pred = None
            if ENCODER: 
                latent = self.env_factor_encoder(privileged_info)
                if DECODER:
                    priv_obs_pred = self.env_factor_decoder(latent)
            else:
                latent = privileged_info
                priv_obs_pred = privileged_info
            actions_mean = self.actor_body(torch.cat((observations, latent), dim=-1))
            policy_info["latents"] = latent.detach().cpu().numpy()
        else:
            actions_mean = self.actor_body(observations)
        return (priv_obs_pred, latent), actions_mean

    def evaluate(self, critic_observations, privileged_observations, **kwargs):
        if USE_LATENT:
            # with torch.no_grad():
            if kwargs['student']:
                if kwargs['observation_history'] is None:
                    raise "No observation history in evaluate" 
                # latent = self.adaptation_module(kwargs['observation_history'])
                latent = self.get_latent_student(kwargs['observation_history'])
            else:
                if ENCODER:
                    latent = self.env_factor_encoder(privileged_observations)
                else:
                    latent = privileged_observations
            value = self.critic_body(torch.cat((critic_observations, latent), dim=-1))
        else:
            value = self.critic_body(critic_observations)
        return value

# class Memory(torch.nn.Module):
#     def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
#         super().__init__()
#         # RNN
#         rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
#         self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
#         self.hidden_states = None
    
#     def forward(self, input, masks=None, hidden_states=None):
#         batch_mode = masks is not None
#         if batch_mode:
#             # batch mode (policy update): need saved hidden states
#             if hidden_states is None:
#                 raise ValueError("Hidden states not passed to memory module during policy update")
#             out, _ = self.rnn(input, hidden_states)
#             out = unpad_trajectories(out, masks)
#         else:
#             # inference mode (collection): use hidden states of last step
#             out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
#         return out
# 
    # def reset(self, dones=None):
    #     # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
    #     for hidden_state in self.hidden_states:
    #         hidden_state[..., dones, :] = 0.0

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
