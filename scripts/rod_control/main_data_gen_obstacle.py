from dataclasses import dataclass
import os
import numpy as np
import os.path as osp
import pickle

from configs import FULL_DIR, NUM_DATA, RANDOM_GRAB_DIR, OBSTACLE_DIR, obstacle_config
from tqdm import tqdm
from ssim.visualize.visualizer import plot_contour_with_spheres_3d
from ssim.utils import load_yaml, save_yaml


@dataclass
class ObstacleConfig:
    num_near_obstacles: int
    num_policy: str
    radius_range: tuple
    x_range: tuple
    y_range: tuple
    z_range: tuple


def get_random_sphere(
    minx: float, maxx: float, miny: float, maxy: float, minz: float,
    maxz: float, radius_range: tuple
) -> tuple:
    radius = np.random.uniform(radius_range[0], radius_range[1])
    x = np.random.uniform(minx + radius, maxx - radius)
    y = np.random.uniform(miny + radius, maxy - radius)
    z = np.random.uniform(minz + radius, maxz - radius)
    return x, y, z, radius


def check_intersection_by_xyz(sphere1: tuple, sphere2: tuple) -> bool:
    x1, y1, z1, r1 = sphere1
    x2, y2, z2, r2 = sphere2
    distance = np.sqrt((x1 - x2)**2 + np.sqrt((y1 - y2)**2) + (z1 - z2)**2)
    return distance <= (r1 + r2)


def check_collision_with_positions(sphere: tuple, positions: np.array) -> bool:
    x, y, z, radius = sphere
    distances = np.sqrt((positions[:, 0] - x)**2 + (positions[:, 1] - y)**2
                        + (positions[:, 2] - z)**2)
    return np.any(distances <= radius)


def check_collision_with_spheres(sphere: tuple, spheres: list) -> bool:
    for other_sphere in spheres:
        if check_intersection_by_xyz(sphere, other_sphere):
            return True
    return False


def gen_obstacles(
    positions: np.array,
    x_range: tuple[float] = None,
    y_range: tuple[float] = None,
    z_range: tuple[float] = None,
    num_obstacles: int | tuple[int] = 5,
    num_policy: str = "random",
    margin: float = 0.5,
    radius_range: tuple = (0.07, 0.1),
) -> list:

    if num_policy == "random":
        num_range = num_obstacles if isinstance(num_obstacles, tuple
                                                ) else (1, num_obstacles + 1)
        num_obstacles = np.random.randint(num_range[0], num_range[1])

    if x_range is None:
        min_x, max_x = positions[:,
                                 0].min() - margin, positions[:,
                                                              0].max() + margin
    else:
        min_x, max_x = x_range
    if y_range is None:
        min_y, max_y = positions[:,
                                 1].min() - margin, positions[:,
                                                              1].max() + margin
    else:
        min_y, max_y = y_range
    if z_range is None:
        min_z, max_z = positions[:,
                                 2].min() - margin, positions[:,
                                                              2].max() + margin
    else:
        min_z, max_z = z_range

    obstacles = []
    for _ in range(num_obstacles):
        while True:
            obstacle = get_random_sphere(
                min_x,
                max_x,
                min_y,
                max_y,
                min_z,
                max_z,
                radius_range,
            )
            if not check_collision_with_positions(
                obstacle, positions
            ) and not check_collision_with_spheres(obstacle, obstacles):
                break
        obstacles.append(obstacle)
    return obstacles


def main():
    script_config = ObstacleConfig(**obstacle_config)
    for i in tqdm(range(NUM_DATA)):
        load_source_dir = osp.join(RANDOM_GRAB_DIR, f"{i}")
        local_full_dir = osp.join(FULL_DIR, f"{i}")
        load_target_dir = osp.join(OBSTACLE_DIR, f"{i}")
        os.makedirs(load_target_dir, exist_ok=True)
        with open(osp.join(local_full_dir, "state_action.pkl"), "rb") as f:
            state_action = pickle.load(f)

        positions = np.array(state_action["rod_position"])
        flattened_positions = positions.transpose(0, 2, 1).reshape(-1, 3)
        obstacles = gen_obstacles(
            flattened_positions,
            script_config.x_range,
            script_config.y_range,
            script_config.z_range,
            script_config.num_near_obstacles,
            script_config.num_policy,
            None,
            script_config.radius_range,
        )

        # export obstacles configs
        base_config = load_yaml(osp.join(load_source_dir, "config.yaml"))
        spheres = []
        for obstacle in obstacles:
            x, y, z, radius = obstacle
            sphere = {
                "type": "sphere",
                "center": [x, y, z],
                "radius": radius,
                "density": 100.0,
            }
            spheres.append(sphere)
        base_config["objects"] = spheres
        save_yaml(base_config, osp.join(load_target_dir, "config.yaml"))

        # visualize
        plot_contour_with_spheres_3d(
            positions=positions.transpose(0, 2, 1),
            spheres=obstacles,
            save_path=osp.join(load_target_dir, "contour.png"),
        )


if __name__ == "__main__":
    np.random.seed(0)
    main()
