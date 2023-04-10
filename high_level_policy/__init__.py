import os

HLP_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
HLP_ENVS_DIR = os.path.join(HLP_ROOT_DIR, 'high_level_policy', 'envs')

PURE_RL_MODEL = f"{HLP_ROOT_DIR}/high_level_policy/models/pure_rl_model"
TEACHER_STUDENT_MODEL = f"{HLP_ROOT_DIR}/high_level_policy/runs/rapid-locomotion/2022-11-29/high_level_train/122321.034489"

USE_LATENT = True

STEP_SIZE = 2
GOAL_THRESHOLD = 0.1
GOAL_POSITION_TRAIN = [3.2, 0]
GOAL_POSITION_VAL = [3.2, 0]
TRAJ_IMAGE_FOLDER = f'traj_push_obs_images_{GOAL_THRESHOLD}_{STEP_SIZE}'
ROLLOUT_HISTORY = 25
EVAL_RATIO = .99
MAX_EPISODE_LENGTH = 15 # 15
ONE_TOUCH_MAP = True
PER_RECT = 7
RECTS = 3
WALLS = True
RESUME_CHECKPOINT = False
SAVE_ADAPTATION_DATA_FILE_NAME = '2obstacle_sim2real/trial_test'


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
    terminal_distance_gs = 5.0
    terminal_ll_reset = 0
    # terminal_pos_reset = -10.0
    terminal_time_out = -5.0 # -5.0
    terminal_gs_velocity = 0

    # step rewards
    distance_gs  = 0.
    velocity_gs = 0.
    distance = -0.0     # 1 - 1/exp(distance to goal)
    time = -0.0 # -0.1
    # action_rate = -0.2
    # ll_reset = -1.0
    # lateral_vel = -0.005
    # angular_vel = -0.005
    # backward_vel = -0.001
    # collision = -0.01
    zero_velocity = -0.2
    # exploration = -0.01
    
    # side_limits = -1.0
    # back_limits = -1.0
    # action_energy = -1.0
    # torque_energy = -1.0
    # base_lin_vel = -0.2
    # base_ang_vel = -0.2


# class reward_scales:
#     # terminal rewards
#     terminal_distance_covered = -0.0
#     terminal_distance_gs = 100.0
#     terminal_ll_reset = 0
#     terminal_time_out = -5.0
#     terminal_gs_velocity = 0

#     # step rewards
#     distance_gs  = 0.
#     velocity_gs = 0.
#     distance = -2.0     # 1 - 1/exp(distance to goal)
#     time = -0.0 # -0.1
#     action_rate = -0.00
#     # ll_reset = -1.0
#     lateral_vel = -0.005
#     angular_vel = -0.005
#     backward_vel = -0.001
#     collision = -0.01
#     zero_velocity = -0.001
#     exploration = 5.0

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
