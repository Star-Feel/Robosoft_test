import os
import os.path as osp
import pickle

import numpy as np
from tqdm import tqdm

from ssim.envs import ControllableGrabArguments, ControllableGrabEnvironment
from ssim.utils import load_json, load_yaml


def main():
    data_path = "./work_dirs/rod_control_data/full/26/info.json"
    work_dir = "./work_dirs/rod_control_demo"
    info = load_json(data_path)
    # Generate data setting
    os.makedirs(work_dir, exist_ok=True)

    # Setting environment
    config_path = info["config"]
    with open(info["state_action"], "rb") as f:
        state_action = pickle.load(f)
    config = load_yaml(config_path)
    object_config = config["objects"][info["object_id"]]
    target_config = config["objects"][info["target_id"]]

    configs = ControllableGrabArguments.from_yaml(config_path)
    env = ControllableGrabEnvironment(configs)
    env.setup()

    update_interval = env.sim_config.update_interval
    total_steps = env.total_steps
    print(
        f"Starting simulation with {total_steps} total steps and update interval {update_interval}..."
    )
    progress_steps = range(0, total_steps, update_interval)
    global_step = 0
    with tqdm(
        total=2 * len(progress_steps), desc="Simulation Progress"
    ) as pbar:

        env.set_target(
            # 指定位置
            np.array(object_config["center"]),
            np.array([0, np.random.uniform(np.pi / 6, np.pi / 3), 0]),
        )

        for i in progress_steps:
            if global_step >= state_action["normal_torque"].shape[0]:
                break

            normal_torque = state_action["normal_torque"][global_step]
            binormal_torque = state_action["binormal_torque"][global_step]
            twist_torque = state_action["twist_torque"][global_step]
            action = np.concatenate(
                (normal_torque, binormal_torque, twist_torque), axis=0
            )
            env.step(action)

            pbar.update(1)
            global_step += 1
            if env.is_achieve(object_config["radius"]):
                print('Rod end point arrives at object.')
                if not any(env.action_flags):
                    env.action_flags[info["object_id"]] = True
                env.set_target(
                    np.array(target_config["center"]),
                    np.array([0, np.random.uniform(np.pi / 6, np.pi / 3), 0]),
                )

            if any(env.action_flags
                   ) and env.is_achieve(object_config["radius"]):
                print('Rod end point arrives at object.')
                break

    # env.visualize_2d(video_name=osp.join(local_output_dir, "2d.mp4"), fps=env.rendering_fps)
    env.visualize_3d(
        video_name=osp.join(work_dir, "3d.mp4"),
        fps=env.rendering_fps,
        xlim=(-0.6, 0.6),
        ylim=(-0.6, 0.6),
        zlim=(0, 1),
    )


if __name__ == "__main__":
    main()
