import copy
import os
import sys
from typing import TypedDict
import numpy as np
import os.path as osp

sys.path.append("/data/zyw/workshop/attempt")
from tqdm import tqdm
from ssim.visualize.visualizer import plot_contour, plot_contour_with_spheres
from ssim.utils import load_yaml, save_yaml, save_json, load_json
import trimesh

N = 100

FULL_DIR = "./work_dirs/navigation_data/full"
TARGET_DIR = "./work_dirs/navigation_data/full"
ASSETS_DIR = "./assets"


class ASSET(TypedDict):
    path: str
    bbox: list


MESH_TEMP = {
    "type": "mesh_surface",
    "mesh_path": "",
    "scale": [],
    "center": [],
    "shape": "",
    "color": "",
}


def get_assets_with_bounding_boxes():
    assets = {}
    for root, dirs, files in os.walk(ASSETS_DIR):
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


def get_random_mesh(assets: dict[ASSET]) -> tuple[str, str, float]:
    mesh_name = np.random.choice(list(assets.keys()))
    mesh_path = assets[mesh_name]["path"]
    mesh_span = max([y - x for x, y in zip(*assets[mesh_name]["bbox"])])
    return str(mesh_name), mesh_path, mesh_span


def main():
    for i in tqdm(range(N)):
        local_full_dir = osp.join(FULL_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        assets = get_assets_with_bounding_boxes()

        config = load_yaml(osp.join(local_full_dir, "config.yaml"))
        spheres = config["objects"]
        for i in range(len(spheres) - 1):
            sphere = spheres[i]
            if sphere["type"] != "sphere":
                continue

            if np.random.rand() < 1 / 3:
                continue

            center = sphere["center"]
            radius = sphere["radius"]

            mesh_name, mesh_path, mesh_span = get_random_mesh(assets)
            mesh_dict = copy.deepcopy(MESH_TEMP)
            mesh_dict["mesh_path"] = mesh_path
            mesh_dict["shape"] = mesh_name
            mesh_dict["center"] = center
            mesh_dict["scale"] = [round(2 * radius / mesh_span, 3)] * 3

            spheres[i] = mesh_dict
        save_yaml(config, osp.join(local_target_dir, "config.yaml"))


if __name__ == "__main__":
    np.random.seed(0)
    main()
