import copy
import os
import sys
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
TARGET_OBJECT_DIR = "./work_dirs/navigation_data/target"
TARGET_DIR = "./work_dirs/navigation_data/full"

info_temp = {
    "id": 0,
    "config": "",
    "state_action": "",
    "visual": "",
    "target_id": 3,
    "description": "Navigation to the green sphere",
}


def main():
    for i in tqdm(range(N)):
        local_random_dir = osp.join(RANDOM_DIR, f"{i}")
        local_target_object_dir = osp.join(TARGET_OBJECT_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        config = load_yaml(osp.join(local_target_object_dir, "config.yaml"))
        spheres = config["objects"]
        for sphere in spheres:
            if "density" not in sphere:
                sphere["density"] = 1.0
            if "position" in sphere:
                sphere["center"] = sphere.pop("position")
        target_id = len(spheres) - 1
        save_yaml(config, osp.join(local_target_dir, "config.yaml"))

        info = copy.deepcopy(info_temp)
        info["id"] = i
        info["config"] = osp.join(local_target_dir, "config.yaml")
        info["state_action"] = osp.join(local_random_dir, "state_action.pkl")
        info["target_id"] = target_id

        save_json(
            info,
            osp.join(local_target_dir, "info.json"),
            indent=4,
            sort_keys=False,
        )


if __name__ == "__main__":
    np.random.seed(0)
    main()
