import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data import CustomDataset, CustomDatasetRNN
from visualization import get_visualization
from model import LSTM, GRU
from tqdm import tqdm
from pathlib import Path

from glob import glob
import pickle
from datetime import datetime

import wandb
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
window_size = 50
sequence_length = 250
hidden_state_size = 1024
num_layers = 2
alg = 'gru'
epochs = 100
train_batch_size = 2500
test_batch_size = train_batch_size
learning_rate = 1e-3
dropout = 0.
print_every = 20
# eval_every = 10
wandb.init(project='indep_model', name=f'{alg}_{sequence_length}_{hidden_state_size}')

SAVE_FOLDER = Path(f'./indep_model/results/{alg}_{sequence_length}_{hidden_state_size}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
PLOT_FOLDER = 'plots'
CHECKPOINT_FOLDER = 'checkpoints'
# data_file = '/common/users/dm1487/legged_manipulation_data/rollout_data/set_3_trajectories/rnn_200k_data.npz'
traj_data_file = '/common/users/dm1487/legged_manipulation_data/rollout_data/latest_individual_traj_mini'
ingore_data_files = '/common/users/dm1487/legged_manipulation_data/rollout_data/ignore_files_latest_individual_traj_mini.pkl'
with open(ingore_data_files, 'rb') as f:
    ignore_files_loaded = pickle.load(f)
idxs_ignored = [int(i.stem.split('_')[-1]) for i in ignore_files_loaded]

# train_idxs = np.concatenate((np.random.randint(0, 60000, 40000), np.random.randint(60000, 120000, 40000), np.random.randint(120000, 180000, 40000)), axis=-1).astype(int).tolist()
all_train_test_files = sorted(glob(f'{traj_data_file}/*.npz'), key=lambda x: int(x.split('.npz')[0].split('/')[-1].split('_')[-1]))
training_size = int(len(all_train_test_files) * 0.9)
val_size = int(len(all_train_test_files) - training_size)

print(training_size, val_size)

train_idxs = np.arange(0, training_size).astype(int).tolist()
# train_idxs = np.arange(training_size//2, (training_size//2)+1000).astype(int).tolist() # for a quick pass through the data
# print(len(train_idxs))
train_idxs = list(set(train_idxs) - set(idxs_ignored))

# train_idxs = np.random.randint(0, 180000, 30000).astype(int).tolist()
val_idxs = [int(i) for i in range(training_size, len(all_train_test_files))]
# val_idxs = np.arange(training_size, training_size+6000).astype(int).tolist() # for a quick pass through the data
# print(len(val_idxs))
val_idxs = list(set(val_idxs) - set(idxs_ignored))


training_files = [all_train_test_files[i] for i in train_idxs]
# val_files = sorted(glob(f'{traj_data_file}/*.npz'), key=lambda x: int(x.split('.npz')[0].split('/')[-1]))
val_files = [all_train_test_files[i] for i in val_idxs]

ds = CustomDatasetRNN(files=training_files, sequence_length=sequence_length, window_size=window_size)
dl = DataLoader(ds, batch_size=train_batch_size, shuffle=True)

test_ds = CustomDatasetRNN(files=val_files, sequence_length=sequence_length, window_size=window_size)
test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)

print(len(ds), len(test_ds))

hidden_states = torch.zeros(num_layers, len(ds), hidden_state_size).to(device)
hidden_states_eval = torch.zeros(num_layers, len(test_ds), hidden_state_size).to(device)

if alg == 'gru':
    model = GRU(input_size=37, hidden_size=hidden_state_size, output_size=14, num_layers=2, dropout=dropout).to(device)
else:
    model = LSTM(input_size=37, hidden_size=hidden_state_size, output_size=14, num_layers=2).to(device)
    cell_states = torch.zeros(num_layers, len(ds), hidden_state_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

def loss_fn(out, targ, mask):
    loss1 = F.binary_cross_entropy(torch.sigmoid(out[:, :, :1]), targ[:, :, :1], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, 7:8]), targ[:, :, 7:8], reduction='none')

    loss2 = F.binary_cross_entropy(torch.sigmoid(out[:, :, 1:2]), targ[:, :, 1:2], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, 8:9]), targ[:, :, 8:9], reduction='none')


    loss3 = F.mse_loss(out[:, :, 2:4], targ[:, :, 2:4], reduction='none') + F.mse_loss(out[:, :, 9:11], targ[:, :, 9:11], reduction='none')
    loss3 = torch.sum(loss3, dim=-1).unsqueeze(-1)

    loss4 = F.mse_loss(out[:, :, 4:7], targ[:, :, 4:7], reduction='none') + F.mse_loss(out[:, :, 11:], targ[:, :, 11:], reduction='none')
    loss4 = torch.sum(loss4, dim=-1).unsqueeze(-1)
    
    return loss1, loss2, loss3, loss4

num_viz = 1
total_loss = 0
current_loss = 0
patches_ctr = 0
for epoch in range(epochs):
    total_loss = 0
    val_idxs = np.random.randint(0, test_batch_size, num_viz)
    model.train()
    train_ctr = 0
    val_ctr = 0
    current_loss = 0
    start = time.time()


    for i, (inp, targ, mask, fsw, idx) in tqdm(enumerate(dl)):
        model.train()
        inp, targ, mask, fsw = inp.to(device), targ.to(device), mask.to(device), fsw.to(device)
        for k in range(0, sequence_length, window_size):
            new_inp, new_mask, new_targ, new_fsw = inp[:, k:k+window_size, :], mask[:, k:k+window_size, :], targ[:, k:k+window_size, :], fsw[:, k:k+window_size, :]
            if alg == 'lstm':
                out, (h, c) = model(new_inp, (hidden_states[:, idx], cell_states[:, idx]))
                hidden_states[:, idx, :], cell_states[:, idx, :] = h.detach(), c.detach()
            elif alg == 'gru':
                out, h = model(new_inp, (hidden_states[:, idx]))
                hidden_states[:, idx, :] = h.detach()

            loss1, loss2, loss3, loss4 = loss_fn(out, new_targ, new_mask)
            loss = loss1 + loss2 + loss3 + loss4
            loss = torch.sum(loss * new_mask) / torch.sum(new_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            total_loss += loss.item()
            train_ctr += 1

            with torch.no_grad():
                wandb.log({
                    'train/contact': (torch.sum(loss1*new_mask)/torch.sum(new_mask)).item(),
                    'train/movable': (torch.sum(loss2*new_mask)/torch.sum(new_mask)).item(),
                    'train/location': (torch.sum(loss3*new_mask)/torch.sum(new_mask)).item(),
                    'train/reconstruction': (torch.sum(loss4*new_mask)/torch.sum(new_mask)).item(),
                })

        if (i+1) % print_every == 0:
            print(f'step {i+1}: {current_loss/print_every} | time: {time.time() - start}')
            current_loss = 0
            start = time.time()

    total_val_loss = 0
    viz_collector = []
    patches = {0: []}
    model.eval()
    anim_idx = np.random.randint(0, test_batch_size)
    with torch.inference_mode():
        for i, (inp, targ, mask, fsw, idx) in tqdm(enumerate(test_dl)):
            inp, targ, mask, fsw = inp.to(device), targ.to(device), mask.to(device), fsw.to(device)
            for k in range(0, sequence_length, window_size):
                new_inp, new_mask, new_targ, new_fsw = inp[:, k:k+window_size, :], mask[:, k:k+window_size, :], targ[:, k:k+window_size, :], fsw[:, k:k+window_size, :]
                if alg == 'lstm':
                    out, (h, c) = model(new_inp, (hidden_states[:, idx], cell_states[:, idx]))
                    hidden_states[:, idx, :], cell_states[:, idx, :] = h.detach(), c.detach()
                elif alg == 'gru':
                    out, h = model(new_inp, (hidden_states[:, idx]))
                    hidden_states[:, idx, :] = h.detach()

                loss1, loss2, loss3, loss4 = loss_fn(out, new_targ, new_mask)
                loss = (loss1 + loss2 + loss3 + loss4)
                loss = torch.sum(loss*new_mask)/torch.sum(new_mask)
                total_val_loss += loss.item()
                with torch.no_grad():
                    wandb.log({
                        'val/contact': (torch.sum(loss1*new_mask)/torch.sum(new_mask)).item(),
                        'val/movable': (torch.sum(loss2*new_mask)/torch.sum(new_mask)).item(),
                        'val/location': (torch.sum(loss3*new_mask)/torch.sum(new_mask)).item(),
                        'val/reconstruction': (torch.sum(loss4*new_mask)/torch.sum(new_mask)).item(),
                    })
                
                if i == 0:
                    for step in range(0, window_size):
                        patches[0].append(get_visualization(anim_idx, new_inp[:, step, :13].squeeze(1), new_targ[:, step, :].squeeze(1), out[:, step, :].squeeze(1), fsw=new_fsw[:, step, :].squeeze(1)))
    scheduler.step(total_val_loss/len(test_dl))
    path = SAVE_FOLDER/f'{PLOT_FOLDER}'
    path.mkdir(parents=True, exist_ok=True)

    for local_idx, (k, v) in enumerate(patches.items()):
        with open(path/f'plot_{patches_ctr+local_idx}.pkl', 'wb') as f:
            pickle.dump(v, f)

    print(f'epoch_loss {epoch}', total_loss/len(dl), f'val_loss {epoch}', total_val_loss/len(test_dl))
    wandb.log({
        'train/loss': total_loss/len(dl),
        'eval/loss': total_val_loss/len(test_dl),
    })

    patches_ctr += num_viz
    total_loss = 0


    # for k in range(0, sequence_length, window_size):
    #     for i, (inp, targ, mask, fsw, done, idx) in tqdm(enumerate(dl)):
    #         hidden_states[:, idx[done], :] = 0.
    #         inp, targ, mask = inp.to(device), targ.to(device), mask.to(device)
            
    #         if alg == 'lstm':
    #             cell_states[:, idx[done], :] = 0.
    #             out, (h, c) = model(inp, (hidden_states[:, idx], cell_states[:, idx]))
    #             hidden_states[:, idx, :], cell_states[:, idx, :] = h.detach(), c.detach()

    #         elif alg == 'gru':
    #             out, h = model(inp, (hidden_states[:, idx]))
    #             hidden_states[:, idx, :] = h.detach()

    #         loss1, loss2, loss3, loss4 = loss_fn(out, targ, mask)
    #         loss = (loss1 + loss2 + loss3 + loss4)
    #         loss = torch.sum(loss*mask)/torch.sum(mask)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         # torch.nn.utils.clip_grad_norm(model.parameters(), 1)
    #         optimizer.step()

    #         train_ctr += 1

    #         total_loss += loss.item()
    #         current_loss += loss.item()
    #         with torch.no_grad():
    #             wandb.log({
    #                 'train/contact': (torch.sum(loss1*mask)/torch.sum(mask)).item(),
    #                 'train/movable': (torch.sum(loss2*mask)/torch.sum(mask)).item(),
    #                 'train/location': (torch.sum(loss3*mask)/torch.sum(mask)).item(),
    #                 'train/reconstruction': (torch.sum(loss4*mask)/torch.sum(mask)).item(),
    #             })
            
    #         if (i+1) % print_every == 0:
    #             print(f'step {i+1}: {current_loss/print_every} | time: {time.time() - start}')
    #             current_loss = 0
    #             start = time.time()
    
            
            
    
                

        # for k in tqdm(range(0, sequence_length, window_size)):
        #     for i, (inp, targ, mask, fsw, done, idx) in enumerate(test_dl):
        #         hidden_states_eval[:, idx[done], :] = 0.
        #         inp, targ, mask, fsw = inp.to(device), targ.to(device), mask.to(device), fsw.to(device)
                
        #         if alg == 'lstm':
        #             cell_states[:, idx[done], :] = 0.
        #             out, (h, c) = model(inp, (hidden_states_eval[:, idx], cell_states[:, idx]))
        #             hidden_states_eval[:, idx, :], cell_states[:, idx, :] = h.detach(), c.detach()
                
        #         elif alg == 'gru':
        #             out, h = model(inp, (hidden_states_eval[:, idx]))
        #             hidden_states_eval[:, idx, :] = h.detach()
               
        #         loss1, loss2, loss3, loss4 = loss_fn(out, targ, mask)
        #         loss = (loss1 + loss2 + loss3 + loss4)
        #         loss = torch.sum(loss*mask)/torch.sum(mask)
        #         total_val_loss += loss.item()


        #         if i == 0:
        #             for step in range(inp.shape[1]):
        #                 patches[0].append(get_visualization(0, inp[:, step, :13].squeeze(1), targ[:, step, :].squeeze(1), out[:, step, :].squeeze(1), fsw=fsw[:, step, :].squeeze(1)))

        #         wandb.log({
        #             'eval/contact': (torch.sum(loss1*mask)/torch.sum(mask)).item(),
        #             'eval/movable': (torch.sum(loss2*mask)/torch.sum(mask)).item(),
        #             'eval/location': (torch.sum(loss3*mask)/torch.sum(mask)).item(),
        #             'eval/reconstruction': (torch.sum(loss4*mask)/torch.sum(mask)).item(),
        #         })

        #         val_ctr += 1

    


    path = SAVE_FOLDER/f'{CHECKPOINT_FOLDER}'
    path.mkdir(parents=True, exist_ok=True)

    # save model every 20 epochs
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), path/f'model_{epoch}.pt')

   
    
    