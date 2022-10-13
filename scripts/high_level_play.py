from ml_logger import logger


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

    actor_critic = ActorCritic(
        num_obs=env.num_obs, 
        num_privileged_obs=env.num_privileged_obs,
        num_obs_history=env.num_obs * \
                        env.num_obs_history,
        num_actions=env.num_actions)

    print(logger.prefix)
    print(logger.glob("*"))
    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    policy = actor_critic.act_inference

    return policy

if __name__ == "__main__":
    from tqdm import tqdm
    from pathlib import Path
    from high_level_policy import HLP_ROOT_DIR
    import glob
    import os
    from high_level_policy.envs.highlevelcontrol import HighLevelControlWrapper 
    from matplotlib import pyplot as plt

    num_envs = 20
    env = HighLevelControlWrapper(num_envs=num_envs, headless=False)

    recent_runs = sorted(glob.glob(f"{HLP_ROOT_DIR}/high_level_policy/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)
    print(recent_runs)
    logger.configure(Path(recent_runs[-1]).resolve())
    policy = load_env(headless=False)

    num_eval_steps = 5000
    obs = env.reset()

    import torch

    for i in tqdm(range(num_eval_steps)):
        # plt.scatter(env.ll_env.all_root_states[:, 0].clone().cpu() - env.ll_env.env_origins[:, 0].clone().cpu(), env.ll_env.all_root_states[:, 1].clone().cpu() - env.ll_env.env_origins[:, 1].clone().cpu() )
        # plt.show()
        # env.reset()
        # actions = env.distance_control()
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, done, info = env.step(actions)