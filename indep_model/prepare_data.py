import numpy as np
from glob import glob
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle


tmp_path = Path(f'/common/users/dm1487/tmp/test_sim2real9')
tmp_path.mkdir(parents=True, exist_ok=True)

data_path = Path('/common/users/dm1487/legged_manipulation_data/rollout_data/random_2obstacle_real/trial_8')
all_pieces = sorted(glob(str(data_path/'*.npz')))[:-1]
data_splits = 1
idxs = np.arange(0, len(all_pieces)//data_splits)
pieces = [all_pieces[i] for i in idxs]
keys = list(np.load(pieces[0]).keys())
one_piece = np.load(pieces[0])
one_piece = np.load(pieces[0])
for key in keys:
    print(key, one_piece[key].shape)

def split_and_pad_trajectories(tensor, dones):
    
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                f]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the input has the following dimension order: [time, number of envs, aditional dimensions]
    """
    # dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    # flat_dones = dones.reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    
    print(trajectory_masks.shape)
    return padded_trajectories, trajectory_masks

def convert_to_traj_and_save(name, tensor_pieces, done_pieces, ctr = 0):

    DATA_PATH_TRAJ = Path(f'/common/users/dm1487/legged_manipulation_data/rollout_data/2_obstacle/trajectories_trial_9/{name}')
    DATA_PATH_TRAJ.mkdir(parents=True, exist_ok=True)
    all_dones = torch.cat(done_pieces, dim=0)
    all_tensor = torch.cat(tensor_pieces, dim=0)
    all_tensor_traj = split_and_pad_trajectories(all_tensor, all_dones)
    if name == 'done_data':
        traj = all_tensor_traj[1].permute(1, 0)
    else:
        traj = all_tensor_traj[0].permute(1, 0, 2)
    start = 0
    offset = 50000
    with tqdm(total=traj.shape[0]) as pbar:
        while True:
            if start >= traj.shape[0]:
                break

            np.savez_compressed(DATA_PATH_TRAJ/f'{name}_{ctr}.npz', data=traj[(start+1):(start+offset-1)])
            start += offset
            pbar.update(offset)
            ctr += 1
    return all_tensor_traj[1], ctr

with open(tmp_path/f'dones.pkl', 'rb') as f:
    d = pickle.load(f)
print(len(d))
# print(d)
for key in [*keys, 'done_data']:
# for key in [*keys[2:3]]:
    a = None
    b = None
    
    print('loading...')
    with open(tmp_path/f'{key}.pkl', 'rb') as f:
        a = pickle.load(f)
    print('done')
    print(len(a))
    
    splits = 5
    start = 0
    offset = len(a)//splits
    ctr = 0
    for _ in range(splits):
        # print(len(d), d[start:])
        _, ctr = convert_to_traj_and_save(key, a[start:(start+offset)], d[start:(start+offset)], ctr)
        start += offset