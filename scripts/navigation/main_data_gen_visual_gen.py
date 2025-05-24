import os
import sys
from turtle import pos
import numpy as np
import os.path as osp
import pickle
import bpy

# sys.path.append("/data/zyw/workshop/attempt")
sys.path.append("/data/wjs/wrp/SoftRoboticaSimulator")

from tqdm import tqdm
from ssim.visualize.visualizer import plot_contour, plot_contour_with_spheres
from ssim.utils import load_yaml, save_yaml, save_json, load_json
from ssim.envs import NavigationSnakeTorqueEnvironment, NavigationSnakeArguments


N = 1

RANDOM_DIR = "/data/wjs/wrp/SoftRoboticaSimulator/test/random_go"
OBSTACLE_DIR = "/data/wjs/wrp/SoftRoboticaSimulator/test/obstacle"
TARGET_DIR = "/data/wjs/wrp/SoftRoboticaSimulator/test/target"

VISUAL_DIR = "/data/wjs/wrp/SoftRoboticaSimulator/test/visual"
FULL_DIR = "/data/wjs/wrp/SoftRoboticaSimulator/test/full"

def main():
    for i in tqdm(range(N)):
        # local_random_dir = osp.join(RANDOM_DIR, f"{i}")
        # local_obstacle_dir = osp.join(OBSTACLE_DIR, f"{i}")
        # local_target_dir = osp.join(TARGET_DIR, f"{i}")

        os.chdir("/data/wjs/wrp/SoftRoboticaSimulator")
        local_full_dir = osp.join(FULL_DIR, f"{i}")
        local_visual_dir = osp.join(VISUAL_DIR, f"{i}")
        os.makedirs(local_visual_dir, exist_ok=True)

        info = load_json(osp.join(local_full_dir, "info.json"))

        # just for test
        info['config'] = info['config'].replace('./work_dirs/navigation_data/', '/data/wjs/wrp/SoftRoboticaSimulator/test/')
        info['state_action'] = info['state_action'].replace('./work_dirs/navigation_data/', '/data/wjs/wrp/SoftRoboticaSimulator/test/')

        config = NavigationSnakeArguments.from_yaml(info["config"])
        env = NavigationSnakeTorqueEnvironment(config)
        env.setup()

        with open(info["state_action"], "rb") as f:
            state_action = pickle.load(f)
        actions = state_action["torque"]

        for i, action in tqdm(enumerate(actions), total=len(actions), desc="Loading state_action"):
            env.step(action)
            if i > 1e5:
                break
            if np.any(env.attach_flags):
                print(env.attach_flags)
                break

            env.single_step_3d_blend(            
                output_images_dir=osp.join(local_visual_dir),
                fps=15,
                width=480,
                height=360,
                current_step=i,
                interval=2000,
            )


        # env.visualize_3d_blender(
        #     video_name='test',
        #     output_images_dir=osp.join(local_visual_dir),
        #     fps=15,
        #     width=480,
        #     height=360,
        # )

        

if __name__ == "__main__":
    np.random.seed(0)
    main()
