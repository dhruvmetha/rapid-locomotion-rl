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

    return actor_critic, policy

if __name__ == "__main__":
    from tqdm import tqdm
    from pathlib import Path
    from high_level_policy import HLP_ROOT_DIR
    import glob
    import os
    from high_level_policy.envs.highlevelcontrol import HighLevelControlWrapper 
    from matplotlib import pyplot as plt

    num_envs = 20
    env = HighLevelControlWrapper(num_envs=num_envs, headless=True, test=False)

    recent_runs = sorted(glob.glob(f"{HLP_ROOT_DIR}/high_level_policy/runs/task_pure_rl/*/*/*"), key=os.path.getmtime)
    # print(recent_runs)
    recent_runs = recent_runs[-1:]
    print(recent_runs)
    logger.configure(Path(recent_runs[-1]).resolve())
    model, policy = load_env(headless=False)

    import torch
    model = model.cpu()
    example = torch.rand(1, 13).cpu()
    actor_model = model.actor_body.cpu()

    traced_script_module = torch.jit.trace(actor_model, example)
    if os.path.exists("./cpp_files") == False:
        os.mkdir("./cpp_files")
    traced_script_module.save("./cpp_files/walking_push_simple_policy_3obs_65_v1.pt")
    print('done')