import copy
import os
import os.path as osp
from dataclasses import dataclass

import bpy
import numpy as np
import tqdm
from main_data_gen_full import desciprtions
from main_data_gen_meshes import (
    get_assets_with_bounding_boxes,
    get_sphere_mesh_data,
    shape2name,
)
from main_data_gen_obstacle import gen_obstacles

from configs import EVAL_DIR, eval_data_config
from ssim.utils import load_yaml, save_json, save_yaml
from ssim.visualize.pov2blend import (
    CameraArguments,
    MeshArguments,
    RodArguments,
    SphereArguments,
    VLNBlenderRenderer,
)


def interpolate_3d(start, direction, base_length, num_points=20):
    """
    在3D空间中从起点沿指定方向均匀插值生成点

    参数:
    start (array-like): 起点坐标 [x, y, z]
    direction (array-like): 方向向量 [dx, dy, dz]
    base_length (float): 总长度
    num_points (int): 生成的点数量，默认为20

    返回:
    numpy.ndarray: 形状为(num_points, 3)的数组，包含所有插值点的坐标
    """
    # 将输入转换为numpy数组
    start = np.array(start, dtype=float)
    direction = np.array(direction, dtype=float)

    # 归一化方向向量
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        raise ValueError("方向向量不能为零向量")

    unit_direction = direction / direction_norm

    # 生成均匀分布的插值参数t
    t_values = np.linspace(0, base_length, num_points)

    # 计算每个t值对应的点坐标
    points = start + t_values.reshape(-1, 1) * unit_direction

    return points


@dataclass
class EvalDataConfig:
    track: str
    num_scanes: int
    num_task_per_scane: int

    base_config_path: str
    base_assets_dir: str
    blend_assets_dir: str

    num_obstacles: int
    num_policy: str
    radius_range: tuple
    canvas_x_range: tuple
    canvas_z_range: tuple

    sphere_probability: float


def main():
    script_config = EvalDataConfig(**eval_data_config)
    total_tasks = 0
    annotations = []
    for _ in tqdm.tqdm(range(script_config.num_scanes)):
        base_config = load_yaml(script_config.base_config_path)
        obstacles = gen_obstacles(
            None,
            script_config.canvas_x_range,
            script_config.canvas_z_range,
            script_config.num_obstacles,
            script_config.num_policy,
            None,
            script_config.radius_range,
            False,
        )
        base_assets = get_assets_with_bounding_boxes(
            script_config.base_assets_dir, "stl"
        )
        blend_assets = get_assets_with_bounding_boxes(
            script_config.blend_assets_dir, "obj"
        )
        objects = []
        for obstacle in obstacles:
            object_dict = get_sphere_mesh_data(
                base_assets,
                blend_assets,
                obstacle,
                script_config.sphere_probability,
            )
            objects.append(object_dict)
        base_config["objects"] = objects

        object_pool = copy.deepcopy(objects)
        for _ in range(script_config.num_task_per_scane):
            if not object_pool:
                break
            local_output_dir = osp.join(
                EVAL_DIR, script_config.track, str(total_tasks)
            )

            os.makedirs(local_output_dir, exist_ok=True)

            # ========== visualization ========== #

            spheres_args = []
            meshes_args = []
            for object_ in base_config["objects"]:
                if object_["type"] == "sphere":
                    sphere_args = SphereArguments(
                        object_["shape"],
                        object_["center"],
                        object_["radius"],
                    )
                    spheres_args.append(sphere_args)
                else:
                    meshes_args.append(
                        MeshArguments(
                            object_["shape"],
                            object_["center"],
                            object_["scale"][0],
                        )
                    )
            rod_config = base_config["rod"]
            positions = interpolate_3d(
                rod_config["start"],
                rod_config["direction"],
                rod_config["base_length"],
                num_points=rod_config["n_elem"],
            ).tolist()
            radius = [rod_config["base_radius"]] * rod_config["n_elem"]
            rod_args = RodArguments(positions, radius, color=(0.45, 0.39, 1))
            blender_renderer = VLNBlenderRenderer(local_output_dir)
            camera_x = (
                max(i[0] for i in obstacles + positions)
                + min(i[0] for i in obstacles + positions)
            ) / 2
            camera_z = (
                max(i[2] for i in obstacles + positions)
                + min(i[2] for i in obstacles + positions)
            ) / 2
            camera_args = CameraArguments(
                [camera_x, 6, camera_z],
                30,
                [camera_x, 0, camera_z],
            )
            blender_renderer.render_by_arguments(
                "image.png",
                camera_args,
                rod_args,
                spheres_args,
                meshes_args,
            )
            # =================================== #

            target_object = np.random.choice(object_pool)
            target_id = base_config["objects"].index(target_object)
            object_pool.remove(target_object)

            instruction = np.random.choice(desciprtions)
            instruction = instruction.replace(
                "<object>", shape2name[target_object["shape"]]
            )
            info = {
                "id": total_tasks,
                "target_id": target_id,
                "instructions": instruction,
            }
            save_yaml(
                base_config, osp.join(
                    local_output_dir,
                    "config.yaml",
                )
            )
            annotations.append(info)

            total_tasks += 1
    save_json(
        annotations,
        osp.join(EVAL_DIR, script_config.track, "annotations.json")
    )


if __name__ == "__main__":
    np.random.seed(0)
    main()
