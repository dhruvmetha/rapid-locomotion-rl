# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]

import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal

from high_level_policy import *
from high_level_policy.utils import unpad_trajectories

# np.random.seed(42)
torch.manual_seed(42)

class RAC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    shared_hidden_dims = [128, 13]
    activation = 'tanh'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [[4096, 2048, 1024, 512, 256, 128, 64]]

    env_factor_encoder_branch_input_dims = [28 if world_cfg.fixed_block.add_to_obs else 9]
    env_factor_encoder_branch_latent_dims = [LATENT_DIM_SIZE if world_cfg.fixed_block.add_to_obs else 4]
    env_factor_encoder_branch_hidden_dims = [[256, 128]]

    env_factor_decoder_branch_input_dims = [LATENT_DIM_SIZE if world_cfg.fixed_block.add_to_obs else 4]
    env_factor_decoder_branch_latent_dims = [28 if world_cfg.fixed_block.add_to_obs else 9]
    env_factor_decoder_branch_hidden_dims = [[256, 128]]


class ActorCritic(nn.Module):
    is_recurrent = True
    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super().__init__()

        activation = get_activation(RAC_Args.activation)

        
        
        total_latent_dim = 0
        if USE_LATENT:
            for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                    zip(RAC_Args.env_factor_encoder_branch_input_dims,
                        RAC_Args.env_factor_encoder_branch_hidden_dims,
                        RAC_Args.env_factor_encoder_branch_latent_dims)):
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

            for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                    zip(RAC_Args.env_factor_decoder_branch_input_dims,
                        RAC_Args.env_factor_decoder_branch_hidden_dims,
                        RAC_Args.env_factor_decoder_branch_latent_dims)):
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

            if not LSTM_ADAPTATION:
                # Adaptation module
                for i, (branch_hidden_dims, branch_latent_dim) in enumerate(zip(RAC_Args.adaptation_module_branch_hidden_dims,
                                                                                RAC_Args.env_factor_encoder_branch_latent_dims)):
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
                class LSTM(nn.Module):
                    def __init__(self, input_size, hidden_size, output_size):
                        super(LSTM, self).__init__()
                        self.hidden_size = hidden_size
                        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                        self.fc = nn.Linear(hidden_size, output_size)

                    def forward(self, observation_history, hidden_states):
                        out, hidden_states = self.lstm(observation_history, hidden_states)
                        out = self.fc(out[:, -1, :])
                        return out, hidden_states

                self.adaptation_module = LSTM(num_obs_history//100, 128, LATENT_DIM_SIZE)
                # self.adaptation_module = nn.Identity()
                self.add_module(f"adaptation_module", self.adaptation_module)

            total_latent_dim += int(torch.sum(torch.Tensor(RAC_Args.env_factor_encoder_branch_latent_dims)))

            # Shared layers
            if SHARED:
                shared_layers = []
                shared_layers.append(nn.Linear(total_latent_dim + num_obs, RAC_Args.shared_hidden_dims[0]))
                shared_layers.append(activation)
                for l in range(len(RAC_Args.shared_hidden_dims)):
                    if l == len(RAC_Args.shared_hidden_dims) - 1:
                        shared_layers.append(nn.Linear(RAC_Args.shared_hidden_dims[l], RAC_Args.shared_hidden_dims[l]))
                    else:
                        shared_layers.append(nn.Linear(RAC_Args.shared_hidden_dims[l], RAC_Args.shared_hidden_dims[l + 1]))
                        shared_layers.append(activation)
                total_latent_dim = 0

        
        # Policy
        self.memory_a = Memory(num_obs+total_latent_dim, type='lstm', num_layers=1, hidden_size=256)
        self.memory_c = Memory(num_obs+total_latent_dim, type='lstm', num_layers=1, hidden_size=256)

        actor_layers = []
        actor_layers.append(nn.Linear(total_latent_dim + num_obs, RAC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(RAC_Args.actor_hidden_dims)):
            if l == len(RAC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(RAC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(RAC_Args.actor_hidden_dims[l], RAC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(total_latent_dim + num_obs, RAC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(RAC_Args.critic_hidden_dims)):
            if l == len(RAC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(RAC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(RAC_Args.critic_hidden_dims[l], RAC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)
        
        # Action noise
        self.std = nn.Parameter(RAC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
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

    def get_latent_student(self, observation_history):
        if not LSTM_ADAPTATION:
            return self.adaptation_module(observation_history)
        else:
            hidden_states = (torch.zeros(1, observation_history.shape[0], 128).to(observation_history.device), torch.zeros(1, observation_history.shape[0], 128).to(observation_history.device))
            observation_history_ = observation_history.view(-1, ROLLOUT_HISTORY, 37)
            return self.adaptation_module(observation_history_, hidden_states)[0]


    def update_distribution(self, observations, privileged_observations, student=False, observation_history=None, hidden_states=None):
        priv_obs_pred_student, priv_obs_pred_teacher = None, None
        latent_teacher, latent_student = None, None
        if USE_LATENT:
            # if student:
            #     if observation_history is None:
            #         raise "No observation history"
                # priv_obs_pred_student = self.env_factor_decoder(latent_student)

            latent_teacher = self.env_factor_encoder(privileged_observations)
            if DECODER:
                priv_obs_pred_teacher = self.env_factor_decoder(latent_teacher)
            
            if student:
                latent_student = self.get_latent_student(observation_history)
                obs_ = self.memory_a(torch.cat((observations, latent_student), dim=-1), hidden_states)
                mean = self.actor_body(obs_)
                # mean = self.actor_body(torch.cat((observations, latent_teacher), dim=-1))
                # if DECODER:
                #     priv_obs_pred_student = self.env_factor_decoder(latent_student)
            else:
                with torch.no_grad():
                    latent_student = self.get_latent_student(observation_history)
                obs_ = self.memory_a(torch.cat((observations, latent_teacher), dim=-1), hidden_states)
                mean = self.actor_body(obs_)
                # mean = self.actor_body(torch.cat((observations, latent_teacher), dim=-1))
                
        else:
            obs_ = self.memory_a(observations, hidden_states)
            mean = self.actor_body(obs_)
            # mean = self.actor_body(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)
        return (priv_obs_pred_teacher, priv_obs_pred_student), (latent_teacher, latent_student)

    def act(self, observations, privileged_observations, **kwargs):
        priv_obs_pred, latent = self.update_distribution(observations, privileged_observations, student=kwargs['student'], observation_history=kwargs['observation_history'], hidden_states=kwargs['hidden_states'])
        return (priv_obs_pred, latent), self.distribution.sample()

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

    def act_student(self, observations, observation_history, hidden_states, policy_info={}):
        priv_obs_pred = None
        latent = None
        if USE_LATENT:
            # print('here', observation_history.shape)
            # latent = self.adaptation_module(observation_history)
            latent = self.get_latent_student(observation_history)
            priv_obs_pred = None
            if DECODER:
                priv_obs_pred = self.env_factor_decoder(latent)

            obs_ = self.memory_a(torch.cat((observations, latent), dim=-1), hidden_states)
            actions_mean = self.actor_body(obs_)
            policy_info["latents"] = latent.cpu().numpy()
        else:
            obs_ = self.memory_a(observations, hidden_states)
            actions_mean = self.actor_body(obs_)
            # actions_mean = self.actor_body(observations)
        return (priv_obs_pred, latent), actions_mean

    def act_teacher(self, observations, privileged_info, hidden_states, policy_info={}):
        priv_obs_pred = None
        latent = None
        if USE_LATENT:
            latent = self.env_factor_encoder(privileged_info)
            priv_obs_pred = None
            if DECODER:
                priv_obs_pred = self.env_factor_decoder(latent)
            obs_ = self.memory_a(torch.cat((observations, latent), dim=-1), hidden_states)
            actions_mean = self.actor_body(obs_)
            
            # actions_mean = self.actor_body(torch.cat((observations, latent), dim=-1))
            policy_info["latents"] = latent.detach().cpu().numpy()
        else:
            obs_ = self.memory_a(observations, hidden_states)
            actions_mean = self.actor_body(obs_)

            # actions_mean = self.actor_body(observations)
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
                latent = self.env_factor_encoder(privileged_observations)
            
            obs_ = self.memory_c(torch.cat((critic_observations, latent), dim=-1), kwargs["hidden_states"])
            value = self.critic_body(obs_)
        else:
            obs_ = self.memory_c(critic_observations, kwargs["hidden_states"])
            value = self.critic_body(obs_)
        return value

class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0

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
