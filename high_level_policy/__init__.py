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
ROLLOUT_HISTORY = 100
OCCUPANCY_GRID = False
STUDENT_ENCODING = 5000
DECODER = True
EVAL_RATIO = .90
MAX_EPISODE_LENGTH = 15
FULL_INFO = True
EVAL_EXPERT = True
ADAPTIVE_SAMPLE_ENVS = False
SHARED = False

experiment_runs = ['pure_rl', 'full_info_decoder', 'teacher_decoder', 'student_decoder', 'teacher_no_decoder', 'student_no_decoder', 'student_decoder_0']
exp = experiment_runs[int(os.environ.get('RUN_TYPE'))]
print(exp)
if exp == 'full_info_decoder':
    FULL_INFO = True
    DECODER = True
    USE_LATENT = True
    EVAL_EXPERT = True
elif exp == 'full_info_no_decoder':
    FULL_INFO = True
    DECODER = False
    USE_LATENT = True
    EVAL_EXPERT = True
elif exp == 'teacher_decoder':
    FULL_INFO = False
    DECODER = True
    USE_LATENT = True
    EVAL_EXPERT = True
elif exp == 'teacher_no_decoder':
    FULL_INFO = False
    DECODER = False
    USE_LATENT = True
    EVAL_EXPERT = True
elif exp == 'pure_rl':
    FULL_INFO = False
    DECODER = False
    USE_LATENT = False
    EVAL_EXPERT = True
elif exp == 'student_decoder':
    FULL_INFO = False
    DECODER = True
    USE_LATENT = True
    EVAL_EXPERT = False
    # STUDENT_ENCODING = 2500
elif exp == 'student_no_decoder':
    FULL_INFO = False
    DECODER = False
    USE_LATENT = True
    EVAL_EXPERT = False
elif exp == 'student_decoder_0':
    FULL_INFO = False
    DECODER = True
    USE_LATENT = True
    EVAL_EXPERT = False
    STUDENT_ENCODING = 0

wandb_config = {
    "project":'legged_navigation', "group":'new_tasks_train_exp_21', "name":f'{exp}', "mode": "online", "notes": "desc: training using a more dense reward, changed zero velocity penalty to 0.2, added task 0 distribution; using easy tasks: task0, task4, task5, task6; deeper adaptation model, entropy coeff: 0.01, observing entropy loss and kl mean, action_rate: 0, velocity penalty scaling : -0.001, history length : 100, addded eval adaptation and reconstruction eval measures, adaptive training, fixed eval set; dataset: same train and test"
}

task_inplay = f'master_task_{exp}'


# RECENT_MODEL = sorted(glob.glob(f"{HLP_ROOT_DIR}/high_level_policy/runs/{task_inplay}/*/*/*"), key=os.path.getmtime)[-1]
# EVAL_MODEL_PATH = RECENT_MODEL

# task_eval_inplay = ['task_1']

# BLOCK



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
    terminal_time_out = -5.0
    terminal_gs_velocity = 0

    # step rewards
    distance_gs  = 0.
    velocity_gs = 0.
    distance = -0.5      # 1 - 1/exp(distance to goal)
    time = -0.0 # -0.1
    action_rate = -0.00
    # ll_reset = -1.0
    lateral_vel = -0.005
    angular_vel = -0.005
    backward_vel = -0.001
    collision = -0.01
    zero_velocity = -0.001


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
