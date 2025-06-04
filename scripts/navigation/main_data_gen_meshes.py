import copy
import os
import os.path as osp
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import trimesh
from tqdm import tqdm

from configs import FULL_DIR, NUM_DATA, meshes_config
from ssim.utils import load_json, load_yaml, save_json, save_yaml


class ASSET(TypedDict):
    path: str
    bbox: list


MESH_TEMP = {
    "type": "mesh_surface",
    "mesh_path": "",
    "scale": [],
    "center": [],
    "shape": "",
}

SPHERE_TEMP = {
    "type": "sphere",
    "density": "",
    "center": [],
    "shape": "",
}


@dataclass
class MeshesConfig:
    base_assets_dir: str
    blend_assets_dir: str


def get_assets_with_bounding_boxes(assets_dir: str, asserts_type: str = "stl"):
    file_appendix = ".stl" if asserts_type == "stl" else ".obj"
    assets = {}
    for root, _, files in os.walk(assets_dir):
        for file in files:
            if file.endswith(file_appendix):
                asset_path = os.path.join(root, file)
                asset_name = osp.splitext(file)[0]

                # Load the mesh and compute the bounding box
                mesh = trimesh.load(asset_path)
                bounding_box = mesh.bounds.tolist(
                )  # Store as a list for JSON compatibility

                # Map asset name to its path and bounding box
                assets[asset_name] = ASSET(path=asset_path, bbox=bounding_box)

    return assets


def get_random_mesh(
    assets: dict[ASSET],
    name: str = None,
    base_mesh_name: str = None
) -> tuple[str, str, float]:
    name_list = list(set(assets.keys()) - set(["Obj", "BasketBall"]))
    if base_mesh_name:
        if base_mesh_name == 'cube':
            name_list = ['pillows_obj', 'Book_by_Peter_Iliev_obj']
        elif base_mesh_name == 'cylinder':
            name_list = ['Coffee_cup_withe_', 'Tea_Cup']
        else:
            name_list = ['conbyfr', 'conbyfr2']

    if name and name in name_list:
        name_list.remove(name)

    mesh_name = np.random.choice(name_list)
    mesh_path = assets[mesh_name]["path"]
    mesh_span = [y - x for x, y in zip(*assets[mesh_name]["bbox"])]
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
    script_config = MeshesConfig(**meshes_config)
    for i in tqdm(range(NUM_DATA)):
        local_full_dir = osp.join(FULL_DIR, f"{i}")
        local_target_dir = osp.join(FULL_DIR, f"{i}")

        os.makedirs(local_target_dir, exist_ok=True)

        base_assets = get_assets_with_bounding_boxes(
            script_config.base_assets_dir, "stl"
        )
        assets = get_assets_with_bounding_boxes(
            script_config.blend_assets_dir, "obj"
        )

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

            (
                mesh_target_name,
                mesh_target_path,
                mesh_target_span,
            ) = get_random_mesh(base_assets)
            blend_mesh_target_name, _, _ = get_random_mesh(
                assets, base_mesh_name=mesh_target_name
            )
            mesh_target_dict = copy.deepcopy(MESH_TEMP)
            mesh_target_dict["mesh_path"] = mesh_target_path
            mesh_target_dict["shape"] = blend_mesh_target_name

            if mesh_target_name == 'cube':
                max_mesh_span = float(np.linalg.norm(mesh_target_span))
            else:
                max_mesh_span = float(max(mesh_target_span))

            scale = round(2 * radius / max_mesh_span, 3)
            mesh_y_height = mesh_target_span[1] * scale
            center[1] = 0.95 * mesh_y_height / 2  # 低一点，保证不会从底下钻过去
            mesh_target_dict["center"] = center
            mesh_target_dict["scale"] = [scale] * 3
            spheres[target_id] = mesh_target_dict

        # 确定除target外的物体信息
        for j in range(len(spheres) - 1):
            if j == target_id:
                continue
            sphere = spheres[j]
            if np.random.rand() < 1 / 3:

                sphere_name, _, _ = get_random_sphere(
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

                mesh_name, mesh_path, mesh_span = get_random_mesh(base_assets)
                blend_mesh_name, _, _ = get_random_mesh(
                    assets, blend_mesh_target_name, mesh_name
                )
                mesh_dict = copy.deepcopy(MESH_TEMP)
                mesh_dict["mesh_path"] = mesh_path
                mesh_dict["shape"] = blend_mesh_name

                if mesh_name == 'cube':
                    max_mesh_span = float(np.linalg.norm(mesh_target_span))
                else:
                    max_mesh_span = float(max(mesh_target_span))

                scale = round(2 * radius / max_mesh_span, 3)
                mesh_y_height = mesh_span[1] * scale
                center[1] = 0.95 * mesh_y_height / 2  # 低一点，保证不会从底下钻过去
                mesh_dict["center"] = center
                mesh_dict["scale"] = [scale] * 3

                spheres[j] = mesh_dict

                del mesh_dict

        # 处理文本
        target_mesh = mesh_target_dict["shape"]
        target_name = ""
        match target_mesh:
            case "Obj":
                target_name = "football"
            case "BasketBall":
                target_name = "basketball"
            case "pillows_obj":
                target_name = "gray pillow"
            case "Book_by_Peter_Iliev_obj":
                target_name = "red book"
            case "Coffee_cup_withe_":
                target_name = "white coffee cup"
            case "Tea_Cup":
                target_name = "tea cup"
            case "conbyfr":
                target_name = "conbyfr"
            case "conbyfr2":
                target_name = "conbyfr2"

        info = load_json(osp.join(local_full_dir, "info.json"))
        info["description"] = info["description"].replace(
            "<object>", target_name
        )
        save_json(info, osp.join(local_target_dir, "info.json"))
        save_yaml(config, osp.join(local_target_dir, "config.yaml"))


if __name__ == "__main__":
    np.random.seed(0)
    main()
