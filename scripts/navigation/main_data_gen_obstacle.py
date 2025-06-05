import os
import os.path as osp
import pickle
from dataclasses import dataclass
from typing import Union

import numpy as np
from tqdm import tqdm

from configs import NUM_DATA, OBSTACLE_DIR, RANDOM_GO_DIR, obstacle_config
from ssim.utils import load_yaml, save_yaml
from ssim.visualize.visualizer import plot_contour_with_spheres


def get_random_sphere(
    minx: float, maxx: float, miny: float, maxy: float, minz: float,
    maxz: float
) -> tuple:
    y = np.random.uniform(miny, maxy)
    radius = y
    x = np.random.uniform(minx + radius, maxx - radius)
    z = np.random.uniform(minz + radius, maxz - radius)
    return x, y, z, radius


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
    for other_sphere in spheres:
        if check_intersection_by_xz(sphere, other_sphere):
            return True
    return False


def gen_obstacles(
    positions: np.array = None,
    x_range: tuple = None,
    z_range: tuple = None,
    num_obstacles: int | tuple[int] = 10,
    num_policy: str = "random",
    margin: float = 0.5,
    radius_range: tuple = (0, 1),
    near: bool = False,
    near_threshold: float = 0.1,
    pre_obsticles: list = None,
) -> list:
    if pre_obsticles is None:
        pre_obsticles = []
    if num_policy == "random":
        num_range = num_obstacles if isinstance(num_obstacles, tuple
                                                ) else (1, num_obstacles + 1)
        num_obstacles = np.random.randint(num_range[0], num_range[1])

    if x_range is None:
        minx, maxx = (
            positions[:, 0].min() - margin,
            positions[:, 0].max() + margin,
        )
    else:
        minx, maxx = x_range
    if z_range is None:
        minz, maxz = (
            positions[:, 2].min() - margin,
            positions[:, 2].max() + margin,
        )
    else:
        minz, maxz = z_range

    obstacles = []
    for _ in range(num_obstacles):
        while True:
            obstacle = get_random_sphere(
                minx, maxx, radius_range[0], radius_range[1], minz, maxz
            )
            if positions is not None:
                position_collision, distance = check_collision_with_positions(
                    obstacle, positions
                )
            else:
                position_collision = False
                distance = 0
            pre_obsticle_collision = check_collision_with_spheres(
                obstacle, pre_obsticles
            )
            obstacle_collision = check_collision_with_spheres(
                obstacle, obstacles
            )

            if position_collision or obstacle_collision \
                    or pre_obsticle_collision:
                continue
            if (not near) or (near and distance < near_threshold):
                break

        obstacles.append(obstacle)
    return obstacles


@dataclass
class ObstacleConfig:
    num_near_obstacles: int
    num_random_obstacles: int
    near_obstacle_num_policy: str
    random_obstacle_num_policy: str
    near_obstacle_radius_range: tuple
    random_obstacle_radius_range: tuple
    near_threshold: float


def main():
    script_config = ObstacleConfig(**obstacle_config)
    for i in tqdm(range(NUM_DATA)):
        load_source_dir = osp.join(RANDOM_GO_DIR, f"{i}")
        load_target_dir = osp.join(OBSTACLE_DIR, f"{i}")
        os.makedirs(load_target_dir, exist_ok=True)
        with open(osp.join(load_source_dir, "state_action.pkl"), "rb") as f:
            state_action = pickle.load(f)

        positions = np.array(state_action["position"])
        flattened_positions = positions[::1000].transpose(0, 2,
                                                          1).reshape(-1, 3)
        near_obstacles = gen_obstacles(
            flattened_positions,
            num_obstacles=script_config.num_near_obstacles,
            num_policy=script_config.near_obstacle_num_policy,
            radius_range=script_config.near_obstacle_radius_range,
            near=True,
            near_threshold=script_config.near_threshold,
        )
        random_obstacles = gen_obstacles(
            flattened_positions,
            num_obstacles=script_config.num_random_obstacles,
            num_policy=script_config.random_obstacle_num_policy,
            radius_range=script_config.random_obstacle_radius_range,
            near=False,
            pre_obsticles=near_obstacles,
        )
        obstacles = near_obstacles + random_obstacles

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
            plot_contour_with_spheres(
                positions=positions.transpose(0, 2, 1)[..., [0, 2]],
                spheres=obstacles,
                save_path=osp.join(load_target_dir, "contour.png"),
            )


if __name__ == "__main__":
    np.random.seed(0)
    main()
