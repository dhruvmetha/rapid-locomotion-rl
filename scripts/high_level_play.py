from ml_logger import logger
from high_level_policy import *

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

if __name__ == "__main__":
    from tqdm import tqdm
    from pathlib import Path
    from high_level_policy import HLP_ROOT_DIR
    import glob
    import os
    from high_level_policy.envs.highlevelcontrol import HighLevelControlWrapper 
    from matplotlib import pyplot as plt

    num_envs = 8  
    env = HighLevelControlWrapper(num_envs=num_envs, headless=False, test=True)

    # recent_runs = sorted(glob.glob(f"{HLP_ROOT_DIR}/high_level_policy/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)
    model_path = EVAL_MODEL_PATH
    logger.configure(Path(model_path).resolve())
    policy_expert, policy_student  = load_env(headless=False)

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

    for i in tqdm(range(num_eval_steps)):
        # plt.scatter(env.ll_env.all_root_states[:, 0].clone().cpu() - env.ll_env.env_origins[:, 0].clone().cpu(), env.ll_env.all_root_states[:, 1].clone().cpu() - env.ll_env.env_origins[:, 1].clone().cpu() )
        # plt.show()
        # env.reset()
        # actions = env.distance_control()
        with torch.no_grad():
            _, actions = policy_student(obs)
        obs, rew, done, info = env.step(actions)
    # print(info['train/episode']['success'])
    print(info['train/success'], info['train/failure'])
    print(info['train/success_rate'])
    print(info['train/ep_length']/info['train/env_count'])