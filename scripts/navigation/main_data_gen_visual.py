import os
import os.path as osp
import pickle
import shutil
from dataclasses import dataclass

import bpy
import numpy as np
from tqdm import tqdm

from configs import FULL_DIR, NUM_DATA, VISUAL_DIR, visualize_config
from ssim.envs import (
    NavigationSnakeArguments,
    NavigationSnakeTorqueEnvironment,
)
from ssim.utils import load_json


@dataclass
class VisualConfig:
    visualize_2d: bool
    width: int
    height: int


def main():
    script_config = VisualConfig(**visualize_config)
    for i in tqdm(range(NUM_DATA)):
        local_full_dir = osp.join(FULL_DIR, f"{i}")
        local_visual_dir = osp.join(VISUAL_DIR, f"{i}")
        shutil.rmtree(local_visual_dir, ignore_errors=True)
        os.makedirs(local_visual_dir, exist_ok=True)

        info = load_json(osp.join(local_full_dir, "info.json"))

        config = NavigationSnakeArguments.from_yaml(info["config"])
        env = NavigationSnakeTorqueEnvironment(config)
        env.setup()

        with open(info["state_action"], "rb") as f:
            state_action = pickle.load(f)
        actions = state_action["torque"]

        for i, action in tqdm(
            enumerate(actions),
            total=len(actions),
            desc="Loading state_action"
        ):
            env.step(action)
            if np.any(env.attach_flags):
                print(env.attach_flags)
                break

        if script_config.visualize_2d:
            env.visualize_2d(
                osp.join(local_visual_dir, "demo.mp4"),
                equal_aspect=True,
                target_last=True,
                fps=env.rendering_fps,
            )
        env.visualize_3d_blender(
            video_name='test',
            output_images_dir=osp.join(local_visual_dir, "visual"),
            fps=env.rendering_fps,
            width=script_config.width,
            height=script_config.height,
        )


if __name__ == "__main__":
    np.random.seed(0)
    main()
