import os
import os.path as osp
import pickle
from dataclasses import dataclass

import numpy as np
import yaml
from tqdm import tqdm

from ssim.envs import (
    NavigationSnakeActionEnvironment,
    NavigationSnakeArguments,
)
from ssim.utils import save_json
from ssim.visualize.visualizer import plot_contour


def run_simulation(
    env: NavigationSnakeActionEnvironment, actions: dict
) -> bool:

    # Get update interval from simulator configuration
    update_interval = env.sim_config.update_interval

    # Run simulation for a number of steps
    total_steps = env.total_steps
    print(
        f"Starting simulation with {total_steps} total steps "
        f"and update interval {update_interval}..."
    )

    # Use tqdm to create a progress bar
    progress_steps = range(0, total_steps, update_interval)
    with tqdm(
        total=total_steps, desc="Simulation Progress", leave=False
    ) as pbar:
        for i in progress_steps:
            if i in actions:
                env.turn[0] = actions[i]
                print(f"Action at step {i}: {actions[i]}")
            env.step()
            pbar.update(1)

    return True


def generate_random_numbers(size: int):
    return np.array(sorted(np.random.choice(range(100), size, replace=False)))


def export_state_action(
    env: NavigationSnakeActionEnvironment, output_dir: str
):
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


def export_config(base_config_path: str, output_dir: str):
    with open(base_config_path, "r") as f:
        env_config = yaml.safe_load(f)
    with open(osp.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(env_config, f, default_flow_style=False)


@dataclass
class RandomGoConfig:
    env_config_path: str
    num_random_actions: int
    visualize: bool


def main():
    # script_config = RandomGoConfig(**random_go_config)
    local_output_dir = "./"

    configs = NavigationSnakeArguments.from_yaml(
        "/data/zyw/workshop/attempt/ssim/configs/random_go.yaml"
    )

    actions = {18000: 0, 33000: 1, 45000: 0}
    env = NavigationSnakeActionEnvironment(configs)
    env.setup(1)
    run_simulation(env, actions)
    env.rod_callback["position"][-1]
    env.visualize_2d(
        osp.join(local_output_dir, "2d.mp4"),
        skip=1000,
        equal_aspect=True,
    )


if __name__ == "__main__":
    np.random.seed(0)
    main()
