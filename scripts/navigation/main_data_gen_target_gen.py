import os
import os.path as osp
import pickle

import numpy as np
from tqdm import tqdm

from ssim.utils import load_yaml, save_yaml
from ssim.visualize.visualizer import plot_contour_with_spheres

N = 100

RANDOM_DIR = "./work_dirs/navigation_data/random_go"
OBSTACLE_DIR = "./work_dirs/navigation_data/obstacle"
TARGET_DIR = "./work_dirs/navigation_data/target"


def gen_target(positions: np.array, scope: float = 0.3):
    maxt = positions.shape[0] - 1
    mint = int(positions.shape[0] * (1 - scope))
    target_t = np.random.randint(mint, maxt)
    target_position = positions[target_t]
    target_radius = np.random.uniform(0.1, 0.5)
    target_position[1] = target_radius
    return *(float(i) for i in target_position), target_radius


def main():
    for i in tqdm(range(N)):
        local_random_dir = osp.join(RANDOM_DIR, f"{i}")
        # local_obstacle_dir = osp.join(OBSTACLE_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        with open(osp.join(local_random_dir, "state_action.pkl"), "rb") as f:
            state_action = pickle.load(f)

        positions = np.array(state_action["position"])
        tip_positions = positions.transpose(0, 2, 1)[:, -1, :]
        tip_positions = tip_positions[::1000]
        target = gen_target(tip_positions, 0.3)

        # export obstacles configs
        base_config = load_yaml(osp.join(local_random_dir, "config.yaml"))
        spheres = []  # base_config["objects"]
        spheres.append({
            "type": "sphere",
            "center": [target[0], target[1], target[2]],
            "radius": float(target[3]),
            "density": 1.0
        })
        base_config["objects"] = spheres
        save_yaml(base_config, osp.join(local_target_dir, "config.yaml"))

        # visualize
        plot_contour_with_spheres(
            positions=positions.transpose(0, 2, 1)[..., [0, 2]],
            spheres=[target],
            save_path=osp.join(local_target_dir, "contour.png"),
        )


if __name__ == "__main__":
    np.random.seed(0)
    main()
