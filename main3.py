import sys

sys.path.append("/data/zyw/workshop/PyElastica")
import os

from tqdm import tqdm

from ssim.envs import GrabMeshArguments, GrabMeshEnvironment
from ssim.utils import is_contact
from ssim.components.surface.mesh_surface import MeshSurface
from ssim.components.contact import RodMeshSurfaceContactWithGridMethod


def run_simulation(env: GrabMeshEnvironment) -> bool:

    # Get update interval from simulator configuration
    update_interval = env.sim_config.update_interval

    # Run simulation for a number of steps
    total_steps = env.total_steps
    print(
        f"Starting simulation with {total_steps} total steps and update interval {update_interval}..."
    )

    # Use tqdm to create a progress bar
    progress_steps = range(0, total_steps, update_interval)
    with tqdm(total=total_steps, desc="Simulation Progress") as pbar:
        counter = 0
        for _ in progress_steps:
            if not any(env.action_flags):
                for idx, object_ in enumerate(env.objects):
                    if is_contact(
                            object_,
                            env.shearable_rod) and idx == 0 and counter == 0:
                        env.action_flags[idx] = True
                        env.uniform_force[-1] = -1
            else:
                counter += 1
            if counter >= 10000:
                for i in range(len(env.action_flags)):
                    env.action_flags[i] = False
                    env.uniform_force[-1] = 0
            env.step()
            pbar.update(1)

    return True


def main():

    config_path = "/data/zyw/workshop/attempt/ssim/configs/rod_mesh.yaml"
    work_dir = "/data/zyw/workshop/attempt/work_dirs"
    os.chdir(work_dir)
    configs = GrabMeshArguments.from_yaml(config_path)

    env = GrabMeshEnvironment(configs)

    env.setup()
    success = run_simulation(env)

    env.visualize_2d(video_name="2d.mp4", fps=env.rendering_fps)
    # env.visualize_3d(video_name="3d.mp4", fps=env.rendering_fps)
    env.export_callbacks("grab_mesh_callbacks.pkl")
    return success


if __name__ == "__main__":
    from stl import mesh
    mesh_data = mesh.Mesh.from_file(
        "/data/zyw/workshop/PyElastica/tests/cube.stl")
    cube = MeshSurface("/data/zyw/workshop/PyElastica/tests/cube.stl")
    import numpy as np
    import elastica as ea
    n_elem = 50
    start = np.zeros((3, ))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 0.35
    base_radius = base_length * 0.011
    density = 1000
    E = 1e6
    poisson_ratio = 0.5
    shear_modulus = E / (poisson_ratio + 1.0)

    rod = ea.CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        base_length,
        base_radius,
        density,
        youngs_modulus=E,
        shear_modulus=shear_modulus,
    )
    mesh_surface = MeshSurface("/data/zyw/workshop/PyElastica/tests/cube.stl")  # 你的网格表面实例
    from ssim.components.contact import surface_grid

    # 定义网格信息
    grid_size = 0.1  # 网格大小
    faces_grid = surface_grid(mesh_data.vectors, grid_size)
    faces_grid["model_path"] = mesh_surface.model_path
    faces_grid["grid_size"] = grid_size
    faces_grid["surface_reorient"] = mesh_surface.mesh_orientation
    k = 1e4  # 接触刚度系数
    nu = 10  # 阻尼系数
    surface_tol = 1e-2  # 表面穿透容差

    # 初始化 RodMeshSurfaceContactWithGridMethod
    contact_method = RodMeshSurfaceContactWithGridMethod(
        k=k,
        nu=nu,
        faces_grid=faces_grid,
        grid_size=grid_size,
        surface_tol=surface_tol,
    )

    # 检查系统有效性（可选）
    contact_method._check_systems_validity(
        system_one=rod,
        system_two=mesh_surface,
        faces_grid=faces_grid,
        grid_size=grid_size,
    )

    # 应用接触力
    out = contact_method.apply_contact(
        system_one=rod,
        system_two=mesh_surface,
    )

    # 输出结果
    main()
