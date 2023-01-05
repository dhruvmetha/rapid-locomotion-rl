import os
import glob

HLP_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# print(HLP_ROOT_DIR)
HLP_ENVS_DIR = os.path.join(HLP_ROOT_DIR, 'high_level_policy', 'envs')

PURE_RL_MODEL = f"{HLP_ROOT_DIR}/high_level_policy/models/pure_rl_model"
TEACHER_STUDENT_MODEL = f"{HLP_ROOT_DIR}/high_level_policy/runs/rapid-locomotion/2022-11-29/high_level_train/122321.034489"

USE_LATENT = True

STEP_SIZE = 2
GOAL_THRESHOLD = 0.1
GOAL_POSITION_TRAIN = [3.2, 0]
GOAL_POSITION_VAL = [3.2, 0]
TRAJ_IMAGE_FOLDER = f'traj_push_obs_images_{GOAL_THRESHOLD}_{STEP_SIZE}'
ROLLOUT_HISTORY = 50
OCCUPANCY_GRID = False
STUDENT_ENCODING = 5000
DECODER = True
EVAL_RATIO = .90
MAX_EPISODE_LENGTH = 15
FULL_INFO = True

experiment_runs = ['full_info_decoder', 'full_info_no_decoder', 'touch_decoder', 'touch_no_decoder', 'pure_rl']
exp = experiment_runs[2]
if exp == 'full_info_decoder':
    FULL_INFO = True
    DECODER = True
    USE_LATENT = True
elif exp == 'full_info_no_decoder':
    FULL_INFO = True
    DECODER = False
    USE_LATENT = True
elif exp == 'touch_decoder':
    FULL_INFO = False
    DECODER = True
    USE_LATENT = True
elif exp == 'touch_no_decoder':
    FULL_INFO = False
    DECODER = False
    USE_LATENT = True
elif exp == 'pure_rl':
    FULL_INFO = False
    DECODER = False
    USE_LATENT = False

wandb_config = {
    "project":'legged_navigation', "group":'random_train_exp_3', "name":exp
}
task_inplay = f'master_task_{exp}'


# RECENT_MODEL = sorted(glob.glob(f"{HLP_ROOT_DIR}/high_level_policy/runs/{task_inplay}/*/*/*"), key=os.path.getmtime)[-1]
# EVAL_MODEL_PATH = RECENT_MODEL

# task_eval_inplay = ['task_1']

# BLOCK
# class world_cfg:
#     CUSTOM_BLOCK = True
#     class movable_block:
#         name = 'movable_block'
#         size_x_range = [0.3, 0.3]
#         size_y_range = [1.8, 1.8] # [0.8, 1.5]
#         pos_x_range = [1.4, 1.5]
#         pos_y_range = [-0.0, 0.0]
#         block_density_range = [1, 6]

#     class fixed_block:
#         add_to_obs = True
#         name = 'fixed_block'
#         num_obs = 2
#         size_x_range = [0.3, 0.3]
#         size_y_range = [0.5, 0.6] # [0.8, 1.5]
#         pos_x_range = [1.8, 1.95]
#         pos_y_range = [-0.6, 0.6]


class world_cfg:
    CUSTOM_BLOCK = True
    class movable_block:
        name = 'movable_block'
        size_x_range = [0.2, 0.4]
        size_y_range = [1.0, 1.0] # [0.8, 1.5]
        pos_x_range = [1.0, 1.5]
        pos_y_range = [-0.0, 0.0]
        block_density_range = [1, 6]

    class fixed_block:
        add_to_obs = True
        name = 'fixed_block'
        num_obs = 2
        size_x_range = [0.1, 0.5]
        size_y_range = [0.3, 0.8] # [0.5, 0.6] # [0.8, 1.5]
        pos_x_range = [1.8, 1.95]
        pos_y_range = [-0.5, 0.5]

class reward_scales:
    # terminal rewards
    terminal_distance_covered = -0.0
    terminal_distance_gs = 100.0
    terminal_ll_reset = 0
    terminal_time_out = 0
    terminal_gs_velocity = 0

    # step rewards
    distance_gs  = 0.
    velocity_gs = 0.
    distance = -0.01      # 1 - 1/exp(distance to goal)
    time = -0.0 # -0.1
    action_rate = -0.0
    # ll_reset = -1.0
    lateral_vel = -0.00
    angular_vel = -0.00
    backward_vel = -0.00
    collision = 0.00


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
