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


class ActorCriticRecurrent(nn.Module):
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
            total_latent_dim = num_privileged_obs
            hidden_size = 256
        # Policy
        self.memory_a = Memory(num_obs+total_latent_dim, type='gru', num_layers=1, hidden_size=hidden_size)
        self.memory_c = Memory(num_obs+total_latent_dim, type='gru', num_layers=1, hidden_size=hidden_size)

        actor_layers = []
        actor_layers.append(nn.Linear(hidden_size, RAC_Args.actor_hidden_dims[0]))
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
        critic_layers.append(nn.Linear(hidden_size, RAC_Args.critic_hidden_dims[0]))
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
   
    def update_distribution(self, observations, privileged_observations, student=False, observation_history=None, hidden_states=None):

        priv_obs_pred_student, priv_obs_pred_teacher = None, privileged_observations
        latent_teacher, latent_student = privileged_observations, None

        if USE_LATENT:
            state = self.memory_a(torch.cat((observations, privileged_observations), dim=-1), hidden_states)
        else:
            state = self.memory_a(observations, hidden_states)

        mean = self.actor_body(state)
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
        priv_obs_pred = privileged_info
        latent = privileged_info
        if USE_LATENT:
            state, next_hidden_states = self.memory_a(torch.cat((observations, privileged_info), dim=-1), hidden_states)
        else:
            state, next_hidden_states = self.memory_a(observations, hidden_states)

        actions_mean = self.actor_body(state)
        return (priv_obs_pred, latent, next_hidden_states), actions_mean


    def evaluate(self, critic_observations, privileged_observations, **kwargs):
        hidden_states = kwargs["hidden_states"]
        if USE_LATENT:
            state, next_hidden_states = self.memory_c(torch.cat((critic_observations, privileged_observations), dim=-1), hidden_states)
        else:
            state, next_hidden_states = self.memory_c(critic_observations, hidden_states)
        
        value = self.critic_body(state)
        return value, next_hidden_states

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
            out, final_hidden_states = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out, final_hidden_states if batch_mode else self.hidden_states

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
