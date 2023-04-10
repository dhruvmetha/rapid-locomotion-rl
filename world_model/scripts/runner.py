import torch

class Runner:
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs

    def run(self, num_iterations=1000):
        # get the observations from the environment
        obs = self.env.reset()
        actions = torch.zeros(self.num_envs, 3)
        # run the environment for 1000 steps
        for j in range(num_iterations):
            # get the actions from the policy
            actions[:, 0] = 1.
            # step the environment
            obs, rewards, dones, info = self.env.step(actions)