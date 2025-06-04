import copy
import os
import os.path as osp
from dataclasses import dataclass
from typing import Union

import numpy as np
from tqdm import tqdm

from configs import (
    FULL_DIR,
    NUM_DATA,
    OBSTACLE_DIR,
    RANDOM_GO_DIR,
    TARGET_DIR,
    VISUAL_DIR,
    full_config,
)
from ssim.utils import load_yaml, save_json, save_yaml

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


def get_description(inplace_object_name: bool = True) -> tuple:
    description = np.random.choice(desciprtions)
    color = ""
    shape = ""
    if inplace_object_name:
        color = np.random.choice(colors).lower()
        shape = np.random.choice(shapes).lower()
        description = description.replace("<object>", f"{color} {shape}")
    return description, color, shape


def check_intersection_by_xz(sphere1: tuple, sphere2: tuple) -> bool:
    x1, _, z1, r1 = sphere1
    x2, _, z2, r2 = sphere2
    distance = np.sqrt((x1 - x2)**2 + (z1 - z2)**2)
    return distance <= (r1 + r2)


def check_collision_with_positions(sphere: tuple,
                                   positions: np.array) -> Union[bool, float]:
    x, _, z, radius = sphere
    distances = np.sqrt((positions[:, 0] - x)**2 + (positions[:, 2] - z)**2)
    return np.any(distances <= radius), np.min(distances - radius)


def check_collision_with_spheres(sphere: tuple, spheres: list) -> bool:
    "True for collision"
    for other_sphere in spheres:
        if check_intersection_by_xz(sphere, other_sphere):
            return True
    return False


@dataclass
class FullConfig:
    inplace_object_name: bool


def main():
    script_config = FullConfig(**full_config)
    for i in tqdm(range(NUM_DATA)):
        local_random_dir = osp.join(RANDOM_GO_DIR, f"{i}")
        local_obstacle_dir = osp.join(OBSTACLE_DIR, f"{i}")
        local_target_object_dir = osp.join(TARGET_DIR, f"{i}")
        local_visual_dir = osp.join(VISUAL_DIR, f"{i}")
        local_target_dir = osp.join(FULL_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        desciprtion, color, shape = get_description(
            script_config.inplace_object_name
        )
        obstacle_config = load_yaml(
            osp.join(local_obstacle_dir, "config.yaml")
        )
        target_config = load_yaml(
            osp.join(local_target_object_dir, "config.yaml")
        )

        assert len(
            target_config["objects"]
        ) == 1, "target config should only have one object"
        obstacle_spheres = obstacle_config["objects"]
        target_spheres = target_config["objects"]
        total_spheres = []
        for sphere in obstacle_spheres:
            center = sphere["center"]
            radius = sphere["radius"]
            sphere_ = (
                float(center[0]),  # x
                float(center[1]),  # y
                float(center[2]),  # z
                float(radius),  # radius
            )
            target_center = target_spheres[0]["center"]
            target_radius = target_spheres[0]["radius"]
            target_sphere_ = (
                float(target_center[0]),  # x
                float(target_center[1]),  # y
                float(target_center[2]),  # z
                float(target_radius),  # radius
            )
            if not check_collision_with_spheres(sphere_, [target_sphere_]):
                total_spheres.append(sphere)
        total_spheres.extend(target_spheres)
        for sphere in total_spheres:
            if "density" not in sphere:
                sphere["density"] = 1.0
            if "position" in sphere:
                sphere["center"] = sphere.pop("position")
        target_id = len(total_spheres) - 1
        if color and shape:
            total_spheres[-1]["color"] = color
            total_spheres[-1]["shape"] = shape
        target_config["objects"] = total_spheres
        save_yaml(target_config, osp.join(local_target_dir, "config.yaml"))

        info = copy.deepcopy(info_temp)
        info["id"] = i
        info["config"] = osp.join(local_target_dir, "config.yaml")
        info["state_action"] = osp.join(local_random_dir, "state_action.pkl")
        info["viusal"] = osp.join(local_visual_dir, "visual")
        info["actions"] = osp.join(local_random_dir, "actions.json")
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
