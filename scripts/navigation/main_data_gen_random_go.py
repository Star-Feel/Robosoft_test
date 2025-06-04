import os
import os.path as osp
import pickle
from dataclasses import dataclass

import numpy as np
import yaml
from tqdm import tqdm

from configs import NUM_DATA, RANDOM_GO_DIR, random_go_config
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
    script_config = RandomGoConfig(**random_go_config)
    output_dir = RANDOM_GO_DIR
    os.makedirs(output_dir, exist_ok=True)

    configs = NavigationSnakeArguments.from_yaml(script_config.env_config_path)

    for i in range(NUM_DATA):
        print(f"Generating data {i + 1}/{NUM_DATA}...")
        local_output_dir = osp.join(output_dir, f"{i}")
        os.makedirs(local_output_dir, exist_ok=True)
        scale = int(
            configs.simulator.final_time / configs.simulator.time_step / 100
        )
        time_steps = scale * generate_random_numbers(
            script_config.num_random_actions
        )
        actions = {}
        last = -1
        for time in time_steps:
            action_list = [0, 1, 2]
            if last > 0:
                action_list.remove(last)
            last = np.random.choice(action_list)
            actions[int(time)] = int(last)
        env = NavigationSnakeActionEnvironment(configs)
        env.setup(1)
        run_simulation(env, actions)

        export_state_action(env, local_output_dir)
        export_config(script_config.env_config_path, local_output_dir)
        save_json(actions, osp.join(local_output_dir, "actions.json"))

        if script_config.visualize:
            env.visualize_2d(
                osp.join(local_output_dir, "2d.mp4"),
                skip=1000,
                equal_aspect=True,
            )
        plot_contour(
            positions=np.array(env.rod_callback["position"]
                               ).transpose(0, 2, 1)[..., [0, 2]],
            save_path=osp.join(local_output_dir, "contour.png")
        )


if __name__ == "__main__":
    np.random.seed(0)
    main()
