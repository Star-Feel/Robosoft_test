import os
import sys
import numpy as np
import os.path as osp
import pickle
import json

import yaml

sys.path.append("/data/zyw/workshop/attempt")
from tqdm import tqdm
from ssim.components import ChangeableMuscleTorques
from ssim.envs import NavigationSnakeActionEnvironment, NavigationSnakeArguments


def run_simulation(env: NavigationSnakeActionEnvironment) -> bool:

    # Get update interval from simulator configuration
    update_interval = env.sim_config.update_interval

    # Run simulation for a number of steps
    total_steps = env.total_steps
    print(
        f"Starting simulation with {total_steps} total steps and update interval {update_interval}..."
    )

    # Use tqdm to create a progress bar
    progress_steps = range(0, total_steps, update_interval)
    with tqdm(total=total_steps, desc="Simulation Progress") as pbar:
        for i in progress_steps:
            if i == 100000:
                print("turn left")
                env.turn[0] = ChangeableMuscleTorques.LEFT
            if i == 200000:
                env.turn[0] = ChangeableMuscleTorques.DIRECT
            if i == 250000:
                env.turn[0] = ChangeableMuscleTorques.RIGHT
            if i == 300000:
                env.turn[0] = ChangeableMuscleTorques.DIRECT
            if i == 500000:
                print("turn right")
                env.turn[0] = ChangeableMuscleTorques.RIGHT
            if i == 600000:
                env.turn[0] = ChangeableMuscleTorques.DIRECT

            env.step()
            pbar.update(1)
            if env.reach(0.05):
                break

    return True


def main():
    output_dir = "./work_dirs/navigation_demo"
    os.makedirs(output_dir, exist_ok=True)

    config_path = "./ssim/configs/navigation_snake.yaml"
    configs = NavigationSnakeArguments.from_yaml(config_path)

    env = NavigationSnakeActionEnvironment(configs)

    env.setup(1)
    env.set_target([0, 0.0, 3.5])
    success = run_simulation(env)

    env_config_path = osp.join(output_dir, "env_config.yaml")
    with open(config_path, "r") as f:
        env_config = yaml.safe_load(f)
    with open(env_config_path, "w") as f:
        yaml.dump(env_config, f)

    torque_callback = env.torque_callback
    rod_callback = env.rod_callback
    state_action = {
        "rod_time": rod_callback["time"],
        "torque_time": np.array([i[0] for i in torque_callback]),
        "position": rod_callback["position"],
        "velocity": rod_callback["velocity"],
        "torque": np.array([i[1] for i in torque_callback]),
    }
    state_action_path = osp.join(output_dir, "state_action.pkl")
    with open(state_action_path, "wb") as f:
        pickle.dump(state_action, f)

    visual_path = osp.join(output_dir, "visual")
    os.makedirs(visual_path, exist_ok=True)
    # env.visualize_2d(video_name=osp.join(visual_path, "2d.mp4"),
    #                  fps=env.rendering_fps)
    # env.visualize_3d_povray(video_name=osp.join(visual_path, "povray.mp4"),
    #                         output_images_dir=visual_path,
    #                         fps=env.rendering_fps)
    info = {
        "id": 0,
        "config": env_config_path,
        "state_action": state_action_path,
        "visual": visual_path,
        "target_id": 3,
        "target_type": "green sphere",
        "description": "Navigation to the green sphere",
    }
    with open(osp.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f)

    return success


if __name__ == "__main__":
    main()
