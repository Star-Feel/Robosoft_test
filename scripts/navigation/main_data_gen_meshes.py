import copy
import os
import os.path as osp
from dataclasses import dataclass
from typing import Optional, Sequence, TypedDict

import numpy as np
import trimesh
from templates import MESH_TEMP, SPHERE_TEMP
from tqdm import tqdm

from configs import FULL_DIR, NUM_DATA, meshes_config
from ssim.utils import load_json, load_yaml, save_json, save_yaml


class ASSET(TypedDict):
    path: str
    bbox: list


@dataclass
class MeshesConfig:
    base_assets_dir: str
    blend_assets_dir: str


shape2name = {
    "Obj": "football",
    "BasketBall": "basketball",
    "pillows_obj": "gray pillow",
    "Book_by_Peter_Iliev_obj": "red book",
    "Coffee_cup_withe_": "white coffee cup",
    "Tea_Cup": "tea cup",
    "conbyfr": "conbyfr",
    "conbyfr2": "conbyfr2",
}


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
    if name is not None:
        name_list.remove(name)

    sphere_name = np.random.choice(name_list)
    sphere_path = assets[sphere_name]["path"]
    sphere_span = max([y - x for x, y in zip(*assets[sphere_name]["bbox"])])
    return str(sphere_name), sphere_path, sphere_span


def get_sphere_mesh_data(
    base_assets: dict[ASSET],
    blend_assets: dict[ASSET],
    base_sphere: dict | Sequence,
    sphere_probability: float = 0.5,
    exclude_asset: Optional[str] = None,
):
    if isinstance(base_sphere, Sequence):
        x, y, z, radius = base_sphere
        base_sphere = SPHERE_TEMP.copy()
        base_sphere["center"] = [x, y, z]
        base_sphere["radius"] = radius

    if np.random.rand() < sphere_probability:
        sphere_target_name, _, _ = get_random_sphere(
            blend_assets, exclude_asset
        )
        object_dict = copy.deepcopy(SPHERE_TEMP)
        object_dict["shape"] = sphere_target_name
        object_dict["density"] = base_sphere['density']
        object_dict["center"] = base_sphere['center']
        object_dict["radius"] = base_sphere['radius']
    else:
        center = base_sphere["center"]
        radius = base_sphere["radius"]

        (
            mesh_target_name,
            mesh_target_path,
            mesh_target_span,
        ) = get_random_mesh(base_assets)
        blend_mesh_target_name, _, _ = get_random_mesh(
            blend_assets,
            exclude_asset,
            mesh_target_name,
        )
        object_dict = copy.deepcopy(MESH_TEMP)
        object_dict["mesh_path"] = mesh_target_path
        object_dict["shape"] = blend_mesh_target_name

        max_mesh_span = float(np.linalg.norm(mesh_target_span))

        scale = round(2 * radius / max_mesh_span, 3)
        mesh_y_height = mesh_target_span[1] * scale
        center[1] = 0.95 * mesh_y_height / 2  # 低一点，保证不会从底下钻过去
        object_dict["center"] = center
        object_dict["scale"] = [scale] * 3
    return object_dict


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

        target_dict = get_sphere_mesh_data(
            base_assets,
            assets,
            spheres[target_id],
            sphere_probability=1 / 2,
            exclude_asset=None
        )

        # 确定除target外的物体信息
        for j in range(len(spheres) - 1):
            if j == target_id:
                continue
            sphere = spheres[j]
            obstacle_dict = get_sphere_mesh_data(
                base_assets,
                assets,
                sphere,
                sphere_probability=1 / 3,
                exclude_asset=target_dict["shape"]
            )

            spheres[j] = obstacle_dict

        # 处理文本
        target_mesh = target_dict["shape"]
        target_name = shape2name.get(target_mesh, target_mesh)

        info = load_json(osp.join(local_full_dir, "info.json"))
        info["description"] = info["description"].replace(
            "<object>", target_name
        )
        save_json(info, osp.join(local_target_dir, "info.json"))
        save_yaml(config, osp.join(local_target_dir, "config.yaml"))


if __name__ == "__main__":
    np.random.seed(0)
    main()
