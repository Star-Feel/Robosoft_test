import os
import os.path as osp
import pickle
import shutil
from dataclasses import dataclass

import bpy
import numpy as np
from tqdm import tqdm

from configs import FULL_DIR, NUM_DATA, VISUAL_DIR, visualize_config
from ssim.envs import ControllableGrabArguments, ControllableGrabEnvironment
from ssim.utils import load_json


@dataclass
class VisualConfig:
    visualize_2d: bool
    width: int
    height: int


def main():
    script_config = VisualConfig(**visualize_config)
    for i in range(NUM_DATA):
        local_full_dir = osp.join(FULL_DIR, f"{i}")
        local_visual_dir = osp.join(VISUAL_DIR, f"{i}")
        shutil.rmtree(local_visual_dir, ignore_errors=True)
        os.makedirs(local_visual_dir, exist_ok=True)

        info = load_json(osp.join(local_full_dir, "info.json"))

        configs = ControllableGrabArguments.from_yaml(info["config"])
        env = ControllableGrabEnvironment(configs)
        env.setup()

        with open(info["state_action"], "rb") as f:
            state_action = pickle.load(f)

        object_id = info["object_id"]
        target_config = configs.objects[info["target_id"]]

        env.set_target_object(object_id)
        env.set_target_point(
            target_config.center,
        )

        update_interval = env.sim_config.update_interval
        total_steps = env.total_steps
        print(
            f"Starting simulation with {total_steps} total steps and "
            f"update interval {update_interval}..."
        )
        progress_steps = range(0, total_steps, update_interval)
        global_step = 0

        with tqdm(
            total=2 * len(progress_steps), desc="Simulation Progress"
        ) as pbar:

            for i in progress_steps:
                if global_step >= state_action["normal_torque"].shape[0]:
                    break

                normal_torque = state_action["normal_torque"][global_step]
                binormal_torque = state_action["binormal_torque"][global_step]
                twist_torque = state_action["twist_torque"][global_step]
                action = np.concatenate(
                    (normal_torque, binormal_torque, twist_torque), axis=0
                )
                action_steps = state_action["action_steps"]
                env.step(action)

                pbar.update(1)
                global_step += 1
                if global_step == action_steps["pick"]:
                    print("pick")
                    env.action_flags[object_id] = True
                elif global_step == action_steps["place"]:
                    break

        env.visualize_3d_blender(
            video_name='test',
            output_images_dir=osp.join(local_visual_dir, 'visual/'),
            fps=15,
            width=script_config.width,
            height=script_config.height,
            target_id=int(info['object_id']),
        )


if __name__ == "__main__":
    main()
