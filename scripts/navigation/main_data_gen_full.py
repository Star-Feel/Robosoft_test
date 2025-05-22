import copy
import os
import os.path as osp

import numpy as np
from tqdm import tqdm

from ssim.utils import load_yaml, save_json, save_yaml

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
desciprtions = [
    "You need to go through the obstacles to find the <object>.",
    "Navigation to the <object>.",
    "Please locate and reach the target: <object>,"
    "and pay attention to obstacles along the way.",
    "Navigate to: <object>, ensuring you avoid "
    "all obstacles to arrive safely.",
    "Your task is to traverse obstacles and successfully locate: <object>.",
    "Explore the environment and find: <object>, "
    "remember to carefully cross any potential obstacles.",
    "Please guide to: <object>, stay alert "
    "during the journey and proceed safely.",
]
colors = [
    "Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Brown",
    "Cyan", "Magenta", "Lime", "Teal", "Indigo", "Violet", "Turquoise",
    "Coral", "Gold", "Silver", "Black", "White"
]
shapes = [
    "Sphere", "Cylinder", "Cube", "Cone", "Pyramid", "Prism", "Torus",
    "Ellipsoid", "Rectangular Prism", "Hexagonal Prism", "Octahedron",
    "Dodecahedron", "Icosahedron", "Tetrahedron", "Parallelepiped",
    "Hemisphere", "Cap", "Frustum", "Disk", "Wedge"
]


def get_description():
    color = np.random.choice(colors).lower()
    shape = np.random.choice(shapes).lower()
    description = np.random.choice(desciprtions)
    description = description.replace("<object>", f"{color} {shape}")
    return description, color, shape


def main():
    for i in tqdm(range(N)):
        local_random_dir = osp.join(RANDOM_DIR, f"{i}")
        local_obstacle_dir = osp.join(OBSTACLE_DIR, f"{i}")
        local_target_object_dir = osp.join(TARGET_OBJECT_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        desciprtion, color, shape = get_description()
        obstacle_config = load_yaml(
            osp.join(local_obstacle_dir, "config.yaml")
        )
        target_config = load_yaml(
            osp.join(local_target_object_dir, "config.yaml")
        )

        assert len(
            target_config["objects"]
        ) == 1, "target config should only have one object"
        spheres = obstacle_config["objects"]
        spheres.extend(target_config["objects"])
        for sphere in spheres:
            if "density" not in sphere:
                sphere["density"] = 1.0
            if "position" in sphere:
                sphere["center"] = sphere.pop("position")
        target_id = len(spheres) - 1
        spheres[-1]["color"] = color
        spheres[-1]["shape"] = shape
        target_config["objects"] = spheres
        save_yaml(target_config, osp.join(local_target_dir, "config.yaml"))

        info = copy.deepcopy(info_temp)
        info["id"] = i
        info["config"] = osp.join(local_target_dir, "config.yaml")
        info["state_action"] = osp.join(local_random_dir, "state_action.pkl")
        info["target_id"] = target_id
        info["description"] = desciprtion

        save_json(
            info,
            osp.join(local_target_dir, "info.json"),
            indent=4,
            sort_keys=False,
        )


if __name__ == "__main__":
    np.random.seed(0)
    main()
