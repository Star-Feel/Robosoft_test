import os
import numpy as np
import os.path as osp
import pickle

from tqdm import tqdm
from ssim.visualize.visualizer import plot_contour, plot_contour_with_spheres
from ssim.utils import load_yaml, save_yaml

Gen_data_Num = 50

SOURCE_DIR = "./work_dirs/rod_control_data/random_go"
FULL_DIR = "./work_dirs/rod_control_data/full"
TARGET_DIR = "./work_dirs/rod_control_data/obstacle"


def get_random_sphere(
    minx: float, maxx: float, miny: float, maxy: float, minz: float,
    maxz: float
) -> tuple:
    radius = np.random.uniform(0, 0.1)
    x = np.random.uniform(minx + radius, maxx - radius)
    y = np.random.uniform(miny + radius, maxy - radius)
    z = np.random.uniform(minz + radius, maxz - radius)
    return x, y, z, radius


def check_intersection_by_xz(sphere1: tuple, sphere2: tuple) -> bool:
    x1, y1, z1, r1 = sphere1
    x2, y2, z2, r2 = sphere2
    distance = np.sqrt((x1 - x2)**2 + (z1 - z2)**2)
    return distance <= (r1 + r2)


def check_collision_with_positions(sphere: tuple, positions: np.array) -> bool:
    x, y, z, radius = sphere
    distances = np.sqrt((positions[:, 0] - x)**2 + (positions[:, 1] - y)**2
                        + (positions[:, 2] - z)**2)
    return np.any(distances <= radius)


def check_collision_with_spheres(sphere: tuple, spheres: list) -> bool:
    for other_sphere in spheres:
        if check_intersection_by_xz(sphere, other_sphere):
            return True
    return False


def gen_obstacles(
    positions: np.array,
    num_obstacles: int = 5,
    num_policy: str = "random",
    margin: float = 0.5
) -> list:

    if num_policy == "random":
        num_obstacles = np.random.randint(1, num_obstacles + 1)
    # minx, maxx = positions[:, 0].min() - margin, positions[:, 0].max() + margin
    # minz, maxz = positions[:, 2].min() - margin, positions[:, 2].max() + margin
    min_x, max_x = -0.6, 0.6
    min_y, max_y = -0.6, 0.6
    min_z, max_z = 0.0, 0.3
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            obstacle = get_random_sphere(
                min_x, max_x, min_y, max_y, min_z, max_z
            )
            if not check_collision_with_positions(
                obstacle, positions
            ) and not check_collision_with_spheres(obstacle, obstacles):
                break
        obstacles.append(obstacle)
    return obstacles


def main():
    for i in tqdm(range(Gen_data_Num)):
        load_source_dir = osp.join(SOURCE_DIR, f"{i}")
        local_full_dir = osp.join(FULL_DIR, f"{i}")
        load_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(load_target_dir, exist_ok=True)
        with open(osp.join(local_full_dir, "state_action.pkl"), "rb") as f:
            state_action = pickle.load(f)

        positions = np.array(state_action["rod_position"])
        flattened_positions = positions.transpose(0, 2, 1).reshape(-1, 3)
        obstacles = gen_obstacles(flattened_positions, num_policy="fix")

        # export obstacles configs
        base_config = load_yaml(osp.join(load_source_dir, "config.yaml"))
        spheres = []
        for obstacle in obstacles:
            x, y, z, radius = obstacle
            sphere = {
                "type": "sphere",
                "center": [x, y, z],
                "radius": radius,
                "density": 1.0,
            }
            spheres.append(sphere)
        base_config["objects"] = spheres
        save_yaml(base_config, osp.join(load_target_dir, "config.yaml"))

        # visualize
        plot_contour_with_spheres(
            positions=positions.transpose(0, 2, 1)[..., [0, 1]],
            spheres=obstacles,
            save_path=osp.join(load_target_dir, "contour.png"),
        )


if __name__ == "__main__":
    np.random.seed(0)
    main()
