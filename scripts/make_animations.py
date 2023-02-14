import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import glob
import os
from tqdm import tqdm 
from pathlib import Path
import time
from high_level_play import task_inplay, RECTS
FFwriter = animation.FFMpegWriter


# RECENT_MODEL = sorted(glob.glob(f"/common/home/dm1487/robotics_research/legged_manipulation/experimental/high_level_policy/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)[-1]
RECENT_MODEL = sorted(glob.glob(f"./high_level_policy/runs/{task_inplay}/*/*/*"), key=os.path.getmtime)[-1]

# RECENT_MODEL = "/common/home/dm1487/robotics_research/legged_manipulation/experimental_bed_2/high_level_policy/runs/task_teacher_decoder/2023-02-13/high_level_train/023242.803256"
source_folder = f"{RECENT_MODEL}/plots_eval"
dest_folder = f"{RECENT_MODEL}/animations_eval"
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)


def on_new_file(tmp_img_path):

    file_name = Path(tmp_img_path).stem
    # if os.path.exists(f"{dest_folder}/{file_name}.mp4"):
    #     continue
    fig, axes = plt.subplots(2, 2, figsize=(48, 24))
    ax = axes.flatten()

    try:

        with open(tmp_img_path, 'rb') as f:
            patches = pickle.load(f)
    except:
        return False
    
    last_patch = []

    def animate(frame):
        if len(last_patch) != 0:
            for i in last_patch:
                try:
                    i.remove()
                except:
                    pass
            last_patch.clear()

        robot, robot_1, robot_2, robot_3 = frame[0], frame[1], frame[2], frame[3]

        ax[0].add_patch(robot)
        ax[1].add_patch(robot_1)
        ax[2].add_patch(robot_2)
        ax[3].add_patch(robot_3)

        ax[0].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='all', aspect='auto')
        ax[1].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='truth', aspect='auto')
        ax[2].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='predicted', aspect='auto')
        ax[3].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='full seen world', aspect='auto')
        
        for i in range(RECTS):
            j = i*6 + 4
            ax[0].add_patch(frame[j])
            ax[0].add_patch(frame[j+1])

            ax[1].add_patch(frame[j+3])
            ax[2].add_patch(frame[j+4])
            ax[3].add_patch(frame[j+5])

        last_patch.extend(frame)
    
    anim = animation.FuncAnimation(fig, animate, frames=patches, interval=10, repeat=False)
    anim.save(f"{dest_folder}/{file_name}.mp4", writer = FFwriter(10))
    plt.close()
    return True

file_list = []

print(source_folder)
while True:
    # get the updated list of files in the folder
    if not os.path.exists(source_folder):
        time.sleep(5)
        continue
    updated_file_list = glob.glob(f"{source_folder}/*.pkl")
    # print(updated_file_list)

    # check for new files
    for file_name in list(reversed(sorted(updated_file_list, key=lambda x: int(Path(x).stem.split('.pkl')[0]))))[:5]:
        if file_name not in file_list:
            done = False
            # call the function when a new file is detected
            while not done:
                print('working on', Path(file_name).stem)
                done = on_new_file(file_name)
                print('done', Path(file_name).stem)

    # update the list of files
    file_list = updated_file_list

    # sleep for a bit before checking again
    time.sleep(1)