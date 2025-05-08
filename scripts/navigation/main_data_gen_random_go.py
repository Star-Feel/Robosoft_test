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
from ssim.visualize.visualizer import plot_contour

N = 100


def run_simulation(env: NavigationSnakeActionEnvironment,
                   actions: dict) -> bool:

    # Get update interval from simulator configuration
    update_interval = env.sim_config.update_interval

    # Run simulation for a number of steps
    total_steps = env.total_steps
    print(
        f"Starting simulation with {total_steps} total steps and update interval {update_interval}..."
    )

    # Use tqdm to create a progress bar
    progress_steps = range(0, total_steps, update_interval)
    with tqdm(total=total_steps, desc="Simulation Progress",
              leave=False) as pbar:
        for i in progress_steps:
            if i in actions:
                env.turn[0] = actions[i]
                print(f"Action at step {i}: {actions[i]}")
            env.step()
            pbar.update(1)

    return True


def generate_random_numbers(size: int):
    return np.array(sorted(np.random.choice(range(100), size, replace=False)))


def export_state_action(env: NavigationSnakeActionEnvironment,
                        output_dir: str):
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


def export_config(config, output_dir: str):

    with open(osp.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def main():
    output_dir = "./work_dirs/navigation_data/random_go"
    os.makedirs(output_dir, exist_ok=True)

    config_path = "./ssim/configs/random_go.yaml"
    configs = NavigationSnakeArguments.from_yaml(config_path)

    for i in range(N):
        print(f"Generating data {i + 1}/{N}...")
        local_output_dir = osp.join(output_dir, f"{i}")
        os.makedirs(local_output_dir, exist_ok=True)
        scale = int(configs.simulator.final_time /
                    configs.simulator.time_step / 100)
        time_steps = scale * generate_random_numbers(6)
        actions = {}
        last = -1
        for time in time_steps:
            action_list = [0, 1, 2]
            if last > 0:
                action_list.remove(last)
            last = np.random.choice(action_list)
            actions[int(time)] = last
        env = NavigationSnakeActionEnvironment(configs)
        env.setup(1)
        run_simulation(env, actions)

        export_state_action(env, local_output_dir)
        with open(config_path, "r") as f:
            env_config = yaml.safe_load(f)
        export_config(env_config, local_output_dir)

        env.visualize_2d(
            osp.join(local_output_dir, "2d.mp4"),
            skip=1000,
            equal_aspect=True,
        )
        plot_contour(positions=np.array(
            env.rod_callback["position"]).transpose(0, 2, 1)[..., [0, 2]],
                     save_path=osp.join(local_output_dir, "contour.png"))


if __name__ == "__main__":
    np.random.seed(0)
    main()
