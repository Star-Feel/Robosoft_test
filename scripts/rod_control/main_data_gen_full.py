import copy
import os
import sys
import numpy as np
import os.path as osp
import random

from tqdm import tqdm
from ssim.utils import load_yaml, save_yaml, save_json

Gen_data_Num = 100

RANDOM_DIR = "./work_dirs/rod_control_data/random_go"
OBSTACLE_DIR = "./work_dirs/rod_control_data/obstacle"
TARGET_OBJECT_DIR = "./work_dirs/rod_control_data/target"
TARGET_DIR = "./work_dirs/rod_control_data/full"

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
    "Place the <object> in the <area> and then place the <object> in the <area>.",
    "Pick up the <object> and place it in the <area>, moving the <object> first.",
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
    obj_color = np.random.choice(colors).lower()
    area_color = np.random.choice(colors).lower()
    shape = np.random.choice(shapes).lower()
    description = np.random.choice(easy_track_desciprtions)
    description = description.replace("<object>", f"{obj_color} {shape}")
    description = description.replace("<area>", f"{area_color} area")
    return description, obj_color, area_color, shape


def main():
    for i in tqdm(range(Gen_data_Num)):
        local_random_dir = osp.join(RANDOM_DIR, f"{i}")
        # local_obstacle_dir = osp.join(OBSTACLE_DIR, f"{i}")
        local_target_object_dir = osp.join(TARGET_OBJECT_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        config = load_yaml(osp.join(local_target_object_dir, "config.yaml"))
        spheres = config["objects"]

        desciprtion, obj_color, tgt_color, obj_shape = get_description()

        for idx, obstacle in enumerate(spheres):
            _, color, _, shape = get_description()
            if obstacle.get("mark", None) == "object":
                color = obj_color
                shape = obj_shape
                object_id = idx
            elif obstacle.get("mark", None) == "target":
                color = tgt_color
                target_id = idx
            obstacle['color'] = color
            obstacle['shape'] = shape
        config["objects"] = spheres
        save_yaml(config, osp.join(local_target_dir, "config.yaml"))

        info = copy.deepcopy(info_temp)
        info["id"] = i
        info["config"] = osp.join(local_target_dir, "config.yaml")
        info["state_action"] = osp.join(local_random_dir, "state_action.pkl")
        info["object_id"] = object_id
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
