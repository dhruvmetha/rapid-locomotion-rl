

import os
import glob


HLP_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# print(HLP_ROOT_DIR)
HLP_ENVS_DIR = os.path.join(HLP_ROOT_DIR, 'high_level_policy', 'envs')
RECENT_MODEL = sorted(glob.glob(f"{HLP_ROOT_DIR}/high_level_policy/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)[-1]
PURE_RL_MODEL = f"{HLP_ROOT_DIR}/high_level_policy/models/pure_rl_model"
TEACHER_STUDENT_MODEL = f"{HLP_ROOT_DIR}/high_level_policy/runs/rapid-locomotion/2022-11-29/high_level_train/122321.034489"

EVAL_MODEL_PATH = RECENT_MODEL
USE_LATENT = True


STEP_SIZE = 2
GOAL_THRESHOLD = 0.1
GOAL_POSITION_TRAIN = [3.5, 0]
GOAL_POSITION_VAL = [3.5, 0]
TRAJ_IMAGE_FOLDER = f'traj_push_obs_images_{GOAL_THRESHOLD}_{STEP_SIZE}'
ROLLOUT_HISTORY = 100

# BLOCK
class world_cfg:
    CUSTOM_BLOCK = True
    class movable_block:
        name = 'movable_block'
        size_x_range = [0.3, 0.3]
        size_y_range = [1.8, 1.8] # [0.8, 1.5]
        pos_x_range = [1.4, 1.5]
        pos_y_range = [-0.0, 0.0]
        block_density_range = [1, 6]

    class fixed_block:
        add_to_obs = True
        name = 'fixed_block'
        num_obs = 2
        size_x_range = [0.3, 0.3]
        size_y_range = [0.5, 0.6] # [0.8, 1.5]
        pos_x_range = [1.8, 1.95]
        pos_y_range = [-0.6, 0.6]

class reward_scales:
    # terminal rewards
    terminal_distance_covered = -0.0
    terminal_distance_gs = 2.0
    terminal_ll_reset = -5.0
    terminal_time_out = -1.0
    terminal_gs_velocity = -2.0

    # step rewards
    distance_gs  = 1.
    velocity_gs = 1.
    distance = -1.0       # 1 - 1/exp(distance to goal)
    time = -0.0 # -0.1
    action_rate = -0.05
    # ll_reset = -1.0
    lateral_vel = -0.025
    angular_vel = -0.025
    backward_vel = -0.025
    collision = -0.05


# def euler_from_quaternion(x, y, z, w):
#     import math
#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     roll_x = math.atan2(t0, t1)
    
#     t2 = +2.0 * (w * y - z * x)
#     t2 = +1.0 if t2 > +1.0 else t2
#     t2 = -1.0 if t2 < -1.0 else t2
#     pitch_y = math.asin(t2)
    
#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw_z = math.atan2(t3, t4)
    
#     return roll_x, pitch_y, yaw_z # in radians