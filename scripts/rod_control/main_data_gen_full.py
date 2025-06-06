import copy
import os
import os.path as osp

import numpy as np
from tqdm import tqdm

from configs import (
    FULL_DIR,
    NUM_DATA,
    OBSTACLE_DIR,
    RANDOM_GRAB_DIR,
    TARGET_DIR,
)
from ssim.utils import load_yaml, save_json, save_yaml

info_temp = {
    "id": 0,
    "config": "",
    "state_action": "",
    "visual": "",
    "object_id": -1,
    "target_id": -1,
    "description": "Navigation to the green sphere",
}

easy_track_desciprtions = [
    "Pick up the <object> and place it in the <area>.",
]

hard_track_desciprtions = [
    "Place the <object> in the <area> and "
    "then place the <object> in the <area>.",
    "Pick up the <object> and place it in the <area>, "
    "moving the <object> first.",
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


def get_description(inplace_object_name: bool = False) -> tuple:

    description = np.random.choice(easy_track_desciprtions)
    if inplace_object_name:
        obj_color = np.random.choice(colors).lower()
        area_color = np.random.choice(colors).lower()
        shape = np.random.choice(shapes).lower()
        description = description.replace("<object>", f"{obj_color} {shape}")
        description = description.replace("<area>", f"{area_color} area")
    else:
        obj_color = None
        area_color = None
        shape = None
    return description, obj_color, area_color, shape


def main():
    for i in tqdm(range(NUM_DATA)):
        local_random_grab_dir = osp.join(RANDOM_GRAB_DIR, f"{i}")
        local_obstacle_dir = osp.join(OBSTACLE_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        local_full_dir = osp.join(FULL_DIR, f"{i}")
        os.makedirs(local_full_dir, exist_ok=True)

        config = load_yaml(osp.join(local_random_grab_dir, "config.yaml"))
        obstacle_config = load_yaml(
            osp.join(local_obstacle_dir, "config.yaml")
        )
        target_object_config = load_yaml(
            osp.join(local_target_dir, "config.yaml")
        )
        spheres = obstacle_config["objects"] + target_object_config["objects"]

        object_id = len(spheres) - 2
        target_id = len(spheres) - 1

        desciprtion, _, _, _ = get_description()

        config["objects"] = spheres
        save_yaml(config, osp.join(local_full_dir, "config.yaml"))

        info = copy.deepcopy(info_temp)
        info["id"] = i
        info["config"] = osp.join(local_full_dir, "config.yaml")
        info["state_action"] = osp.join(local_full_dir, "state_action.pkl")
        info["object_id"] = object_id
        info["target_id"] = target_id
        info["description"] = desciprtion

        save_json(
            info,
            osp.join(local_full_dir, "info.json"),
            indent=4,
            sort_keys=False,
        )


if __name__ == "__main__":
    np.random.seed(0)
    main()
