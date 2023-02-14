import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader 
import numpy as np
from glob import glob
import torch
from data import CustomDataset
from model import MiniTransformer
from tqdm import tqdm

import wandb
from pathlib import Path
from datetime import datetime
from visualization import get_visualization
import pickle
from config import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
window_size = 250
sequence_length = 250
hidden_state_size = 2048
num_layers = 8
alg = 'transformer'
eval_every = 250
print_every = 50
epochs = 100
train_batch_size = 128
test_batch_size = 128
learning_rate = 1e-4
dropout = 0.
input_size = 25
output_size = 21
# RECTS = 2

wandb.init(project='indep_model', name=f'{alg}_{sequence_length}_{hidden_state_size}_3obs/')

SAVE_FOLDER = Path(f'./indep_model/results/{alg}_{sequence_length}_{hidden_state_size}_3obs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
PLOT_FOLDER = 'plots'
CHECKPOINT_FOLDER = 'checkpoints'

#### works really well
# traj_data_file = '/common/users/dm1487/legged_manipulation_data/rollout_data/latest_individual_traj_mini'
# ingore_data_files = '/common/users/dm1487/legged_manipulation_data/rollout_data/ignore_files_latest_individual_traj_mini.pkl'
# input_size = 37
# output_size = 14
# RECTS = 2
# learning_rate = 1e-4
####

#### 3 obstacle
traj_data_file = '/common/users/dm1487/legged_manipulation_data/rollout_data/3_obstacle/traj_list_train_bb_traj'
ingore_data_files = '/common/users/dm1487/legged_manipulation_data/rollout_data/3_obstacle/ignore_random_traj_bb.pkl'
label_data_files = '/common/users/dm1487/legged_manipulation_data/rollout_data/3_obstacle/labels_random_traj_bb.pkl'
#####

with open(ingore_data_files, 'rb') as f:
    ignore_files_loaded = pickle.load(f)
with open(label_data_files, 'rb') as f:
    label_idx = pickle.load(f)

#     pickle.dump(label_idx, f)

idxs_ignored = [int(i.stem.split('_')[-1]) for i in ignore_files_loaded]

idxs_valid = {1: [], 2:[], 3:[]}
train_labels = []
val_labels = []
for k, v in label_idx.items():
    idxs_valid[k] = [int(i.stem.split('_')[-1]) for i in list(set(v) - set(ignore_files_loaded))]
limit_class = min(len(idxs_valid[1][:-2000]), len(idxs_valid[2][:-2000]), len(idxs_valid[3][:-2000]))
train_idxs_valid_all = []
for k, v in idxs_valid.items():
    train_labels.extend((np.zeros(limit_class) + k).astype(int).tolist())
    train_idxs_valid_all.extend(np.random.choice(v[:-2000], limit_class))
val_idxs_valid_all = []
for k, v in idxs_valid.items():
    val_labels.extend((np.zeros(2000) + k).astype(int).tolist())
    val_idxs_valid_all.extend(np.random.choice(v[-2000:], 2000))

# idxs_ignored =[]
# train_idxs = np.concatenate((np.random.randint(0, 60000, 40000), np.random.randint(60000, 120000, 40000), np.random.randint(120000, 180000, 40000)), axis=-1).astype(int).tolist()
all_train_test_files = sorted(glob(f'{traj_data_file}/*.npz'), key=lambda x: int(x.split('.npz')[0].split('/')[-1].split('_')[-1]))
# training_size = int(len(all_train_test_files) * 0.9)
# val_size = int(len(all_train_test_files) - training_size)

# print(training_size, val_size)

# train_idxs = np.arange(0, training_size).astype(int).tolist()
# # print(len(train_idxs))
# train_idxs = list(set(train_idxs) - set(idxs_ignored))
# print(len(train_idxs))
# train_idxs = [int(i) for i in np.random.randint(0, len(train_idxs), 10000).astype(int).tolist()]
# train_idxs = np.random.randint(0, 180000, 30000).astype(int).tolist()
# val_idxs = [int(i) for i in range(training_size, len(all_train_test_files))]
# # print(len(val_idxs))
# val_idxs = list(set(val_idxs) - set(idxs_ignored))
# print(len(val_idxs))
# val_idxs = np.random.randint(1000, 1300, 300).astype(int).tolist()
# print(len(train_idxs))
# val_idxs = train_idxs # [int(i) for i in np.random.randint(0, len(val_idxs), 1000).astype(int).tolist()]

training_files = [all_train_test_files[i] for i in train_idxs_valid_all]
# val_files = sorted(glob(f'{traj_data_file}/*.npz'), key=lambda x: int(x.split('.npz')[0].split('/')[-1]))
val_files = [all_train_test_files[i] for i in val_idxs_valid_all]

train_ds = CustomDataset(files=training_files, labels = train_labels, input_size=input_size, sequence_length=sequence_length, window_size=window_size)
train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)

val_ds = CustomDataset(files=val_files, labels = val_labels, input_size=input_size, sequence_length=sequence_length, window_size=window_size)
val_dl = DataLoader(val_ds, batch_size=train_batch_size, shuffle=True)

print(len(train_ds), len(val_ds))

model = MiniTransformer(input_size=input_size, output_size=output_size, embed_size=128, hidden_size=hidden_state_size, num_heads=8, max_sequence_length=250, num_layers=num_layers)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dl), epochs=epochs)

def loss_fn(out, targ, mask):
    per_rect = 7
    extract_idx = [torch.arange(per_rect*0, per_rect*1, dtype=torch.long, device=device), torch.arange(per_rect*1, per_rect*2, dtype=torch.long, device=device), torch.arange(per_rect*2, per_rect*3, dtype=torch.long, device=device), torch.arange(per_rect*3, per_rect*4, dtype=torch.long, device=device)]
    reconstruction_loss = None

    loss1 = F.binary_cross_entropy(torch.sigmoid(out[:, :, :1]), targ[:, :, :1], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, 7:8]), targ[:, :, 7:8], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, 14:15]), targ[:, :, 14:15], reduction='none') # + F.binary_cross_entropy(torch.sigmoid(out[:, :, 21:22]), targ[:, :, 21:22], reduction='none')

    loss2 = F.binary_cross_entropy(torch.sigmoid(out[:, :, 1:2]), targ[:, :, 1:2], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, 8:9]), targ[:, :, 8:9], reduction='none') + F.binary_cross_entropy(torch.sigmoid(out[:, :, 15:16]), targ[:, :, 15:16], reduction='none') # + F.binary_cross_entropy(torch.sigmoid(out[:, :, 22:23]), targ[:, :, 22:23], reduction='none')


    loss3 = F.mse_loss(out[:, :, 2:4], targ[:, :, 2:4], reduction='none') + F.mse_loss(out[:, :, 9:11], targ[:, :, 9:11], reduction='none') + F.mse_loss(out[:, :, 16:18], targ[:, :, 16:18], reduction='none') # + F.mse_loss(out[:, :, 23:25], targ[:, :, 23:25], reduction='none')
    loss3 = torch.sum(loss3, dim=-1).unsqueeze(-1)

    loss4 = F.mse_loss(out[:, :, 4:7], targ[:, :, 4:7], reduction='none') + F.mse_loss(out[:, :, 11:14], targ[:, :, 11:14], reduction='none') + F.mse_loss(out[:, :, 18:21], targ[:, :, 18:21], reduction='none') # + F.mse_loss(out[:, :, 25:], targ[:, :, 25:], reduction='none')
    loss4 = torch.sum(loss4, dim=-1).unsqueeze(-1)
    
    return loss1, loss2, loss3, loss4

src_mask = torch.triu(torch.ones(250, 250) * float('-inf'), diagonal=1).to(device)
all_anim = []
patches_ctr = 0
for epoch in range(epochs):
    train_total_loss = 0
    current_train_loss = 0
    model.train()
    for i, (inp, targ, mask, fsw, done, idx) in tqdm(enumerate(train_dl)):
        inp = inp.to(device)
        targ = targ.to(device)
        mask = mask.to(device)

        # print('tr', inp.shape, targ.shape, mask.shape, fsw.shape)

        # new_mask = torch.ones_like(mask) * float('-inf')
        # new_mask[mask.nonzero(as_tuple=True)] = 0.
        # new_mask = new_mask.to(device)

        out = model(inp, src_mask=src_mask)

        loss1, loss2, loss3, loss4 = loss_fn(out, targ, mask)
        loss = (loss1 + loss2 + loss3 + loss4)
        # print(out.shape, loss.shape, mask.shape)


        loss = torch.sum(loss*mask)/torch.sum(mask)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        optimizer.step()

        current_train_loss += loss.item()
        train_total_loss += loss.item()

        if (i+1)%print_every == 0:
            print(f'step {i+1}:', current_train_loss/print_every)
            current_train_loss = 0

        with torch.no_grad():
            wandb.log({
                'train/contact': (torch.sum(loss1*mask)/torch.sum(mask)).item(),
                'train/movable': (torch.sum(loss2*mask)/torch.sum(mask)).item(),
                'train/location': (torch.sum(loss3*mask)/torch.sum(mask)).item(),
                'train/reconstruction': (torch.sum(loss4*mask)/torch.sum(mask)).item(),
            })

        anim_ctr = 0 

        if (i+1) % eval_every == 0:
            val_total_loss = 0
            anim_idx = np.random.randint(0, test_batch_size)
            with torch.inference_mode():
                model.eval()
                for i, (inp, targ, mask, fsw, done, idx) in tqdm(enumerate(val_dl)):
                    inp = inp.to(device)
                    targ = targ.to(device)
                    mask = mask.to(device)
                    fsw = fsw.to(device)

                    anim_idx = min(anim_idx, inp.shape[0]-1)

                    out = model(inp, src_mask=src_mask)

                    loss1, loss2, loss3, loss4 = loss_fn(out, targ, mask)
                    loss = (loss1 + loss2 + loss3 + loss4)
                    loss = torch.sum(loss*mask)/torch.sum(mask)

                    val_total_loss += loss.item()

                    wandb.log({
                            'eval/contact': (torch.sum(loss1*mask)/torch.sum(mask)).item(),
                            'eval/movable': (torch.sum(loss2*mask)/torch.sum(mask)).item(),
                            'eval/location': (torch.sum(loss3*mask)/torch.sum(mask)).item(),
                            'eval/reconstruction': (torch.sum(loss4*mask)/torch.sum(mask)).item(),
                    })
                    if anim_ctr < 10:
                        patches = []
                        # print(inp.shape[1])
                        for step in range(inp.shape[1]):
                            # print(mask[anim_idx, step, 0])
                            if mask[anim_idx, step, 0]:
                                patch_set = get_visualization(anim_idx, inp[:, step, :13].squeeze(1), targ[:, step, :].squeeze(1), out[:, step, :].squeeze(1), fsw[:, step, :].squeeze(1))
                                patches.append(patch_set)
                        all_anim.append(patches)
                        anim_ctr += 1
            
            path = SAVE_FOLDER/f'{PLOT_FOLDER}'
            path.mkdir(parents=True, exist_ok=True)

            # print(len(all_anim), len(all_anim[0]))

            for local_idx, anim in enumerate(all_anim):
                # print(len(anim))
                with open(path/f'plot_{patches_ctr+local_idx}.pkl', 'wb') as f:
                    pickle.dump(anim, f)
            patches_ctr += len(all_anim)
            all_anim = []

            scheduler.step(val_total_loss)


            print(f'Epoch: {epoch}, Loss: {train_total_loss/eval_every}, Val Loss: {val_total_loss/len(val_dl)}')
            wandb.log({
                'train/loss': train_total_loss/eval_every,
                'eval/loss': val_total_loss/len(val_dl)
            })

            train_total_loss = 0

    path = SAVE_FOLDER/f'{CHECKPOINT_FOLDER}'
    path.mkdir(parents=True, exist_ok=True)

    # save model every 20 epochs
    if (epoch+1) % 1 == 0:
        torch.save(model.state_dict(), path/f'model_{epoch}.pt')