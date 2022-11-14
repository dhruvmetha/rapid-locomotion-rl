

import os

HLP_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
HLP_ENVS_DIR = os.path.join(HLP_ROOT_DIR, 'high_level_policy', 'envs')
USE_LATENT = False

STEP_SIZE = 2
GOAL_THRESHOLD = 0.1
GOAL_POSITION_TRAIN = [3, 0]
GOAL_POSITION_VAL = [3, 0]
TRAJ_IMAGE_FOLDER = f'traj_push_obs_images_{GOAL_THRESHOLD}_{STEP_SIZE}'


# BLOCK
class world_cfg:
    CUSTOM_BLOCK = True
    class movable_block:
        name = 'movable_block'
        size_x_range = [0.3, 0.8]
        size_y_range = [0.3, 0.9] # [0.8, 1.5]
        pos_x_range = [1.0, 2.25]
        pos_y_range = [-0.5, 0.5]
        block_density_range = [1, 5]

    class fixed_block:
        add_to_obs = False
        name = 'fixed_block'
        num_obs = 2
        size_x_range = [0.3, 0.3]
        size_y_range = [0.5, 0.5] # [0.8, 1.5]
        pos_x_range = [2.1, 2.1]
        pos_y_range = [-0.6, 0.6]

class reward_scales:
    # terminal rewards
    terminal_distance_covered = -0.01
    terminal_distance_gs = 2.0
    terminal_ll_reset = -1.0
    terminal_time_out = -1.0
    terminal_gs_velocity = -2.0

    # step rewards
    distance_gs  = 0.5
    velocity_gs = 0.2
    distance = -1.0  # 1 - 1/exp(distance to goal)
    time = -0.0 # -0.1
    action_rate = -0.02
    # ll_reset = -1.0
    lateral_vel = -0.25
    angular_vel = -0.15
    backward_vel = -0.15
    collision = -1.0