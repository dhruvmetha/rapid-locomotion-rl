import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import glob
import os
from tqdm import tqdm 
from pathlib import Path
FFwriter = animation.FFMpegWriter

RECENT_MODEL = sorted(glob.glob(f"/home/dhruv/projects_dhruv/priv_blind_run/high_level_policy/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)[-1]
source_folder = f"{RECENT_MODEL}/plots_eval"
dest_folder = f"{RECENT_MODEL}/animations_eval"
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

files = sorted(glob.glob(f"{source_folder}/*.pkl"))

for idx, tmp_img_path in tqdm(enumerate(files)):
    print(tmp_img_path)

    file_name = Path(tmp_img_path).stem
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))

    with open(tmp_img_path, 'rb') as f:
        patches = pickle.load(f)
        last_patch = []

        def animate(frame):
            if len(last_patch) != 0:
                for i in last_patch:
                    try:
                        i.remove()
                    except:
                        pass
                last_patch.clear()
            
            robot, robot_1, robot_2 = frame[0], frame[1], frame[2]

            ax[0].add_patch(robot)
            ax[1].add_patch(robot_1)
            ax[2].add_patch(robot_2)

            ax[0].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='all')
            ax[1].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='truth')
            ax[2].set(xlim=(-1.0, 4.0), ylim=(-1, 1), title='predicted')
            
            for i in range(3):
                j = i*4 + 3
                ax[0].add_patch(frame[j])
                ax[0].add_patch(frame[j+1])

                ax[1].add_patch(frame[j+2])
                ax[2].add_patch(frame[j+3])

            last_patch.extend(frame)
        
        anim = animation.FuncAnimation(fig, animate, frames=patches, interval=10, repeat=False)
        anim.save(f"{dest_folder}/{file_name}.mp4", writer = FFwriter(30))
        plt.close()