import os
import sys
import numpy as np
import os.path as osp
import pickle

sys.path.append("/data/zyw/workshop/attempt")
from tqdm import tqdm
from ssim.visualize.visualizer import plot_contour, plot_contour_with_spheres
from ssim.utils import load_yaml, save_yaml, save_json, load_json

N = 100

SOURCE_DIR = "./work_dirs/navigation_data/random_go"
TARGET_DIR = "./work_dirs/navigation_data/obstacle"


def get_random_sphere(minx: float, maxx: float, miny: float, maxy: float,
                      minz: float, maxz: float) -> tuple:
    y = np.random.uniform(miny, maxy)
    radius = y
    x = np.random.uniform(minx + radius, maxx - radius)
    z = np.random.uniform(minz + radius, maxz - radius)
    return x, y, z, radius


def check_intersection_by_xz(sphere1: tuple, sphere2: tuple) -> bool:
    x1, y1, z1, r1 = sphere1
    x2, y2, z2, r2 = sphere2
    distance = np.sqrt((x1 - x2)**2 + (z1 - z2)**2)
    return distance <= (r1 + r2)


def check_collision_with_positions(sphere: tuple, positions: np.array) -> bool:
    x, y, z, radius = sphere
    distances = np.sqrt((positions[:, 0] - x)**2 + (positions[:, 2] - z)**2)
    return np.any(distances <= radius)


def check_collision_with_spheres(sphere: tuple, spheres: list) -> bool:
    for other_sphere in spheres:
        if check_intersection_by_xz(sphere, other_sphere):
            return True
    return False


def gen_obstacles(positions: np.array,
                  num_obstacles: int = 10,
                  num_policy: str = "random",
                  margin: float = 0.5) -> list:

    if num_policy == "random":
        num_obstacles = np.random.randint(1, num_obstacles + 1)
    minx, maxx = positions[:, 0].min() - margin, positions[:, 0].max() + margin
    minz, maxz = positions[:, 2].min() - margin, positions[:, 2].max() + margin
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            obstacle = get_random_sphere(minx, maxx, 0, 1, minz, maxz)
            if not check_collision_with_positions(
                    obstacle, positions) and not check_collision_with_spheres(
                        obstacle, obstacles):
                break
        obstacles.append(obstacle)
    return obstacles


def main():
    for i in tqdm(range(N)):
        load_source_dir = osp.join(SOURCE_DIR, f"{i}")
        load_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(load_target_dir, exist_ok=True)
        with open(osp.join(load_source_dir, "state_action.pkl"), "rb") as f:
            state_action = pickle.load(f)

        positions = np.array(state_action["position"])
        flattened_positions = positions.transpose(0, 2, 1).reshape(-1, 3)
        obstacles = gen_obstacles(flattened_positions, num_policy="random")

        # export obstacles configs
        base_config = load_yaml(osp.join(load_source_dir, "config.yaml"))
        spheres = []
        for obstacle in obstacles:
            x, y, z, radius = obstacle
            sphere = {
                "type": "sphere",
                "position": [x, y, z],
                "radius": radius,
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
