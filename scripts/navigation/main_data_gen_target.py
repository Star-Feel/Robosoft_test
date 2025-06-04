import os
import os.path as osp
import pickle
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from configs import NUM_DATA, RANDOM_GO_DIR, TARGET_DIR, target_config
from ssim.utils import load_yaml, save_yaml
from ssim.visualize.visualizer import plot_contour_with_spheres


def gen_target(
    positions: np.array,
    scope: float = 0.3,
    radius_range: tuple = (0.1, 0.5),
) -> tuple:
    maxt = positions.shape[0] - 1
    mint = int(positions.shape[0] * (1 - scope))
    target_t = np.random.randint(mint, maxt)
    target_position = positions[target_t]
    target_radius = np.random.uniform(*radius_range)
    target_position[1] = target_radius
    return *(float(i) for i in target_position), target_radius


@dataclass
class TargetConfig:
    scope: float
    radius_range: tuple


def main():
    script_config = TargetConfig(**target_config)
    for i in tqdm(range(NUM_DATA)):
        local_random_dir = osp.join(RANDOM_GO_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        with open(osp.join(local_random_dir, "state_action.pkl"), "rb") as f:
            state_action = pickle.load(f)

        positions = np.array(state_action["position"])
        tip_positions = positions.transpose(0, 2, 1)[:, -1, :]
        tip_positions = tip_positions[::1000]
        target = gen_target(
            tip_positions,
            script_config.scope,
            script_config.radius_range,
        )

        # export obstacles configs
        base_config = load_yaml(osp.join(local_random_dir, "config.yaml"))
        spheres = []  # base_config["objects"]
        spheres.append({
            "type": "sphere",
            "center": [target[0], target[1], target[2]],
            "radius": float(target[3]),
            "density": 100.0
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
