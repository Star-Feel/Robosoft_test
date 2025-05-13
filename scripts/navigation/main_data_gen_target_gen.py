import os
import sys
from turtle import pos
import numpy as np
import os.path as osp
import pickle

sys.path.append("/data/zyw/workshop/attempt")
from tqdm import tqdm
from ssim.visualize.visualizer import plot_contour, plot_contour_with_spheres
from ssim.utils import load_yaml, save_yaml, save_json, load_json

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
    return target_position[0], target_position[1], target_position[
        2], target_radius


def main():
    for i in tqdm(range(N)):
        local_random_dir = osp.join(RANDOM_DIR, f"{i}")
        local_obstacle_dir = osp.join(OBSTACLE_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        with open(osp.join(local_random_dir, "state_action.pkl"), "rb") as f:
            state_action = pickle.load(f)

        positions = np.array(state_action["position"])
        tip_positions = positions.transpose(0, 2, 1)[:, -1, :]
        tip_positions = tip_positions[::100]
        target = gen_target(tip_positions, 0.3)

        # export obstacles configs
        base_config = load_yaml(osp.join(local_obstacle_dir, "config.yaml"))
        spheres = base_config["objects"]
        spheres.append({
            "type": "sphere",
            "position": list(target[:3]),
            "radius": target[3],
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
