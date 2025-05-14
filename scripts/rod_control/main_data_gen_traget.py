import os
import os.path as osp
import pickle

import numpy as np
from tqdm import tqdm

from ssim.utils import load_yaml, save_yaml

N = 100

RANDOM_DIR = "./work_dirs/rod_control_data/random_go"
OBSTACLE_DIR = "./work_dirs/rod_control_data/obstacle"
TARGET_DIR = "./work_dirs/rod_control_data/target"


def get_object(
    position: np.ndarray,
    direction: np.array,
    radius_range: tuple[float] = (0, 0.1)
):
    radius = np.random.uniform(radius_range[0], radius_range[1])
    center = position + direction * radius
    return float(radius), center.tolist()


def get_target(
    position: np.ndarray,
    direction: np.array,
    object_radius: float,
):
    target_radius, target_center = get_object(
        position, direction, (object_radius, object_radius)
    )
    target_center[-1] -= target_radius
    return target_radius, target_center


def main():
    for i in tqdm(range(N)):
        local_random_dir = osp.join(RANDOM_DIR, f"{i}")
        local_obstacle_dir = osp.join(OBSTACLE_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        with open(osp.join(local_random_dir, "state_action.pkl"), "rb") as f:
            state_action = pickle.load(f)

        positions = state_action["rod_position"].transpose(0, 2, 1)
        process_steps = state_action["process_steps"]
        time_shots = state_action["time_shots"]
        # Find the index of a specific element in the np.ndarray
        pick = np.argwhere(process_steps == time_shots[0])[0][0]
        place = np.argwhere(process_steps == time_shots[1])
        place = place[0][0] if place.size > 0 else -1

        pick_position = positions[pick, -1, :]
        pick_direction = positions[pick, -1, :] - positions[pick, -2, :]
        pick_direction = pick_direction / np.linalg.norm(pick_direction)
        place_position = positions[place, -1, :]
        place_direction = positions[place, -1, :] - positions[place, -2, :]
        place_direction = place_direction / np.linalg.norm(place_direction)
        object_radius, object_center = get_object(
            pick_position, pick_direction, (0, 0.1)
        )
        target_radius, target_center = get_target(
            place_position, place_direction, object_radius
        )

        # export obstacles configs
        base_config = load_yaml(osp.join(local_obstacle_dir, "config.yaml"))
        spheres = base_config["objects"]
        spheres.append({
            "type": "sphere",
            "center": object_center,
            "radius": object_radius,
            "density": 1.0,
            "mark": "object"
        })
        spheres.append({
            "type": "sphere",
            "center": target_center,
            "radius": target_radius,
            "density": 1.0,
            "mark": "target"
        })
        save_yaml(base_config, osp.join(local_target_dir, "config.yaml"))


if __name__ == "__main__":
    np.random.seed(0)
    main()
