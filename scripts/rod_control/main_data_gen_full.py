import copy
import os
import sys
import numpy as np
import os.path as osp
import random

sys.path.append("/data/wjs/SoftRoboticaSimulator")
from tqdm import tqdm
from ssim.utils import load_yaml, save_yaml, save_json

Gen_data_Num = 2

RANDOM_DIR = "/data/wjs/SoftRoboticaSimulator/work_dirs/rod_control_data/random_go"
OBSTACLE_DIR = "/data/wjs/SoftRoboticaSimulator/work_dirs/rod_control_data/obstacle"
# TARGET_OBJECT_DIR = "./work_dirs/rod_control_data/target"
TARGET_DIR = "/data/wjs/SoftRoboticaSimulator/work_dirs/rod_control_data/full"

info_temp = {
    "id": 0,
    "config": "",
    "state_action": "",
    "visual": "",
    "target_id": 3,
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
        local_obstacle_dir = osp.join(OBSTACLE_DIR, f"{i}")
        # local_target_object_dir = osp.join(TARGET_OBJECT_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        # config = load_yaml(osp.join(local_target_object_dir, "config.yaml"))
        desciprtion, obj_color, area_color, shape = get_description()
        config = load_yaml(osp.join(local_random_dir, "config.yaml"))
        obj_spheres = config["objects"]

        # obj
        obj_spheres[0]['color'] = obj_color
        obj_spheres[0]['type'] = shape

        # target
        obj_spheres[1]['color'] = area_color
        # spheres[1]['type'] = shape

        obstacle_config = load_yaml(osp.join(local_obstacle_dir, "config.yaml"))
        obstacle_spheres = obstacle_config["objects"]

        for obstacle in obstacle_spheres:
            _, obj_color, _, shape = get_description()
            obstacle['color'] = obj_color
            obstacle['type'] = shape

        spheres = obj_spheres + obstacle_spheres
        random.shuffle(spheres)

        target_id = spheres.index(obj_spheres[1])
        
        config["objects"] = spheres
        save_yaml(config, osp.join(local_target_dir, "config.yaml"))

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
        break


if __name__ == "__main__":
    np.random.seed(0)
    main()
