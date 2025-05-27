import copy
import os
import os.path as osp
from typing import TypedDict

import numpy as np
import trimesh
from tqdm import tqdm
import sys

sys.path.append("/data/wjs/wrp/SoftRoboticaSimulator")

from ssim.utils import load_yaml, save_yaml, load_json

N = 100

FULL_DIR = "./work_dirs/navigation_data/full"
TARGET_DIR = "./work_dirs/navigation_data/full"
BASE_ASSETS_DIR = "./assets"
ASSETS_DIR = "./scene_assets/living_room"

# FULL_DIR = "/data/wjs/wrp/SoftRoboticaSimulator/test/full"
# TARGET_DIR = "/data/wjs/wrp/SoftRoboticaSimulator/test/full"
# BASE_ASSETS_DIR = "/data/wjs/wrp/SoftRoboticaSimulator/assets"
# ASSETS_DIR = "/data/wjs/wrp/SoftRoboticaSimulator/scene_assets/living_room"


class ASSET(TypedDict):
    path: str
    bbox: list


MESH_TEMP = {
    "type": "mesh_surface",
    "mesh_path": "",
    "blend_mesh_path": "",
    "scale": [],
    "center": [],
    "shape": "",
}

SPHERE_TEMP = {
    "type": "sphere",
    "density": "",
    "blend_sphere_path": "",
    "center": [],
    "shape": "",
}


def get_base_assets_with_bounding_boxes():
    assets = {}
    for root, _, files in os.walk(BASE_ASSETS_DIR):
        for file in files:
            if file.endswith(".stl"):
                asset_path = os.path.join(root, file)
                asset_name = osp.splitext(file)[0]

                # Load the mesh and compute the bounding box
                mesh = trimesh.load(asset_path)
                bounding_box = mesh.bounds.tolist(
                )  # Store as a list for JSON compatibility

                # Map asset name to its path and bounding box
                assets[asset_name] = ASSET(path=asset_path, bbox=bounding_box)

    return assets


def get_assets_with_bounding_boxes():
    assets = {}
    for root, _, files in os.walk(ASSETS_DIR):
        for file in files:
            # if file.endswith(".stl"):
            #     asset_path = os.path.join(root, file)
            #     asset_name = osp.splitext(file)[0]

            #     # Load the mesh and compute the bounding box
            #     mesh = trimesh.load(asset_path)
            #     bounding_box = mesh.bounds.tolist(
            #     )  # Store as a list for JSON compatibility

            #     # Map asset name to its path and bounding box
            #     assets[asset_name] = ASSET(path=asset_path, bbox=bounding_box)

            if file.endswith(".obj"):
                asset_path = os.path.join(root, file)
                asset_name = osp.splitext(file)[0]

                # Load the mesh and compute the bounding box
                mesh = trimesh.load(asset_path)
                bounding_box = mesh.bounds.tolist(
                )  # Store as a list for JSON compatibility

                # Map asset name to its path and bounding box
                assets[asset_name] = ASSET(path=asset_path, bbox=bounding_box)

    return assets


def get_random_mesh(assets: dict[ASSET],
                    name: str = None) -> tuple[str, str, float]:
    name_list = list(set(assets.keys()) - set(["Obj", "BasketBall"]))
    if name:
        name_list.remove(name)

    mesh_name = np.random.choice(name_list)
    mesh_path = assets[mesh_name]["path"]
    mesh_span = max([y - x for x, y in zip(*assets[mesh_name]["bbox"])])
    return str(mesh_name), mesh_path, mesh_span


def get_random_sphere(assets: dict[ASSET],
                      name: str = None) -> tuple[str, str, float]:
    name_list = list(["Obj", "BasketBall"])
    if name:
        name_list.remove(name)

    sphere_name = np.random.choice(name_list)
    sphere_path = assets[sphere_name]["path"]
    sphere_span = max([y - x for x, y in zip(*assets[sphere_name]["bbox"])])
    return str(sphere_name), sphere_path, sphere_span


def main():
    for i in tqdm(range(N)):
        local_full_dir = osp.join(FULL_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")

        os.makedirs(local_target_dir, exist_ok=True)

        base_assets = get_base_assets_with_bounding_boxes()
        assets = get_assets_with_bounding_boxes()

        config = load_yaml(osp.join(local_full_dir, "config.yaml"))
        spheres = config["objects"]

        info = load_json(osp.join(FULL_DIR, f"{i}", "info.json"))
        target_id = info["target_id"]

        sphere_target_name = None
        blend_mesh_target_name = None

        # 确定target信息
        if np.random.rand() < 1 / 2:
            sphere_target_name, _, _ = get_random_sphere(assets)
            sphere_target_dict = copy.deepcopy(SPHERE_TEMP)
            sphere_target_dict["shape"] = sphere_target_name
            sphere_target_dict["density"] = spheres[target_id]['density']
            sphere_target_dict["center"] = spheres[target_id]['center']
            sphere_target_dict["radius"] = spheres[target_id]['radius']
            spheres[target_id] = sphere_target_dict

        else:
            center = spheres[target_id]["center"]
            radius = spheres[target_id]["radius"]

            _, mesh_target_path, mesh_target_span = get_random_mesh(
                base_assets
            )
            blend_mesh_target_name, blend_mesh_target_path, _ = get_random_mesh(
                assets
            )
            mesh_target_dict = copy.deepcopy(MESH_TEMP)
            mesh_target_dict["mesh_path"] = mesh_target_path
            mesh_target_dict["shape"] = blend_mesh_target_name
            mesh_target_dict["center"] = center
            mesh_target_dict["scale"] = [
                round(2 * radius / mesh_target_span, 3)
            ] * 3
            spheres[target_id] = mesh_target_dict

        # 确定除target外的物体信息
        for j in range(len(spheres) - 1):
            if j == target_id:
                continue
            sphere = spheres[j]
            if np.random.rand() < 1 / 3:

                sphere_name, sphere_path, sphere_span = get_random_sphere(
                    assets, sphere_target_name
                )
                sphere_dict = copy.deepcopy(SPHERE_TEMP)
                sphere_dict["shape"] = sphere_name
                sphere_dict["density"] = sphere['density']
                sphere_dict["center"] = sphere['center']
                sphere_dict["radius"] = sphere['radius']
                spheres[j] = sphere_dict

                del sphere_dict

            else:
                center = sphere["center"]
                radius = sphere["radius"]

                _, mesh_path, mesh_span = get_random_mesh(base_assets)
                blend_mesh_name, blend_mesh_path, _ = get_random_mesh(
                    assets, blend_mesh_target_name
                )
                mesh_dict = copy.deepcopy(MESH_TEMP)
                mesh_dict["mesh_path"] = mesh_path
                mesh_dict["shape"] = blend_mesh_name
                mesh_dict["center"] = center
                mesh_dict["scale"] = [round(2 * radius / mesh_span, 3)] * 3

                spheres[j] = mesh_dict

                del mesh_dict

        save_yaml(config, osp.join(local_target_dir, "config.yaml"))


if __name__ == "__main__":
    np.random.seed(0)
    main()
