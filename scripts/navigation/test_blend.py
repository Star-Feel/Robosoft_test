from dataclasses import dataclass

import bpy
import numpy as np

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

    # ========== visualization ========== #

    sphere_args = [
        SphereArguments(
            "Obj",
            [-0.5, 0.5, 0],
            0.5,
        ),
        SphereArguments(
            "BasketBall",
            [-0.5, 0.5, -1],
            0.5,
        )
    ]
    meshes_args = [
        MeshArguments(
            "conbyfr",
            [-1, 0, 1],
            1,
        ),
        MeshArguments(
            "conbyfr2",
            [0, 0, 1],
            1,
        ),
        MeshArguments(
            "Tea_Cup",
            [1, 0, 1],
            1,
        ),
        MeshArguments(
            "Book_by_Peter_Iliev_obj",
            [0.5, 0, 0],
            1,
        ),
        MeshArguments(
            "pillows_obj",
            [0.5, 0, -1],
            1,
        ),
        MeshArguments(
            "Coffee_cup_withe_",
            [0.5, 0, 1.7],
            1,
        ),
    ]
    positions = interpolate_3d(
        [0, 0, 0],
        [0, 0, 1],
        0.5,
        num_points=20,
    ).tolist()
    radius = [0.05] * 20
    rod_args = RodArguments(positions, radius, color=(0.45, 0.39, 1))
    blender_renderer = VLNBlenderRenderer("./")
    camera_x = 0
    camera_z = 0
    camera_args = CameraArguments(
        [camera_x, 6, camera_z],
        30,
        [camera_x, 0, camera_z],
    )
    blender_renderer.render_by_arguments(
        "image.png",
        camera_args,
        rod_args,
        sphere_params=sphere_args,
        mesh_params=meshes_args,
    )
    # =================================== #


if __name__ == "__main__":
    np.random.seed(0)
    main()
