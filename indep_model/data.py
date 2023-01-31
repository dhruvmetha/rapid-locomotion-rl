from torch.utils.data import Dataset
import numpy as np
from glob import glob
import torch

class CustomDataset(Dataset):
    def __init__(self, files, input_size, sequence_length, window_size=25):
        # self.folder = folder
        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.window_size = int(window_size)
        self.input_size = int(input_size)
        self.batch_sequence_segment = np.zeros((len(self.all_folders)), dtype=np.int)

    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        # if idx == 0:
        #     print(self.batch_sequence_segment[idx])
        data = np.load(self.all_folders[idx])
        inp, target, mask, fsw = torch.tensor(data['obs_hist']), torch.tensor(data['priv_obs']), torch.tensor(data['done']), torch.tensor(data['fsw'])

        inp_idx, targ_idx, mask_idx, fsw_idx = inp[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size], \
                            target[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size], \
                            mask[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size], \
                            fsw[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size]
        # if self.batch_sequence_segment[idx] == inp.shape[0] - self.window_size:
        if self.batch_sequence_segment[idx] == self.sequence_length - self.window_size:
            self.batch_sequence_segment[idx] = 0
        else:
            self.batch_sequence_segment[idx] += self.window_size
        return inp_idx[:, :self.input_size], targ_idx, mask_idx, fsw_idx, self.batch_sequence_segment[idx] == self.window_size, idx


class CustomDatasetRNN(Dataset):
    def __init__(self, files, sequence_length, window_size=25):
        # self.folder = folder
        self.all_folders = files
        self.sequence_length = int(sequence_length)
        self.window_size = int(window_size)
        self.batch_sequence_segment = np.zeros((len(self.all_folders)), dtype=np.int)

    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        # if idx == 0:
        #     print(self.batch_sequence_segment[idx])
        data = np.load(self.all_folders[idx])
        inp, target, mask, fsw = torch.tensor(data['obs_hist']), torch.tensor(data['priv_obs']), torch.tensor(data['done']), torch.tensor(data['fsw'])

        return inp, target, mask, fsw, idx

        inp_idx, targ_idx, mask_idx, fsw_idx = inp[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size], \
                            target[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size], \
                            mask[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size], \
                            fsw[self.batch_sequence_segment[idx]:self.batch_sequence_segment[idx]+self.window_size]
        # if self.batch_sequence_segment[idx] == inp.shape[0] - self.window_size:
        if self.batch_sequence_segment[idx] == self.sequence_length - self.window_size:
            self.batch_sequence_segment[idx] = 0
        else:
            self.batch_sequence_segment[idx] += self.window_size
        return inp_idx, targ_idx, mask_idx, fsw_idx, self.batch_sequence_segment[idx] == self.window_size, idx

