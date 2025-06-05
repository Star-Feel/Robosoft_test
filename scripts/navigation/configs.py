WORK_DIR = "/data/zyw/workshop/attempt/work_dirs/navigation_data_new"

NUM_DATA = 100
RANDOM_GO_DIR = f"{WORK_DIR}/random_go"
OBSTACLE_DIR = f"{WORK_DIR}/obstacle"
TARGET_DIR = f"{WORK_DIR}/target"
VISUAL_DIR = f"{WORK_DIR}/visual"
FULL_DIR = f"{WORK_DIR}/full"
RELEASE_DIR = f"{WORK_DIR}/release"
EVAL_DIR = f"{WORK_DIR}/eval"

random_go_config = {
    "env_config_path": "./ssim/configs/random_go.yaml",
    "num_random_actions": 3,
    "visualize": True,
}

prune_config = {
    "prune_rate": 0.5,
}

obstacle_config = {
    "num_near_obstacles": 5,
    "near_obstacle_num_policy": "fix",
    "near_obstacle_radius_range": (0.1, 0.5),
    "near_threshold": 0.1,
    "num_random_obstacles": 5,
    "random_obstacle_num_policy": "fix",
    "random_obstacle_radius_range": (0.1, 0.3),
}

target_config = {
    "scope": 0.1,
    "radius_range": (0.1, 0.5),
}

full_config = {
    "inplace_object_name": False,
}

meshes_config = {
    "base_assets_dir": "./assets",
    "blend_assets_dir": "./scene_assets/living_room",
}
visualize_config = {
    "visualize_2d": True,
    "width": 480,
    "height": 360,
}

release_config = {}

# eval_data_config = {
#     "track": "easy",
#     "num_scanes": 100,
#     "num_task_per_scane": 3,
#     "base_config_path": "./ssim/configs/random_go.yaml",
#     "base_assets_dir": "./assets",
#     "blend_assets_dir": "./scene_assets/living_room",
#     "num_obstacles": 5,
#     "num_policy": "random",
#     "radius_range": (0.1, 0.5),
#     "canvas_x_range": (-1.5, 1.5),
#     "canvas_z_range": (0, 3),
#     "sphere_probability": 0.3,
# }

eval_data_config = {
    "track": "hard",
    "num_scanes": 100,
    "num_task_per_scane": 3,
    "base_config_path": "./ssim/configs/random_go.yaml",
    "base_assets_dir": "./assets",
    "blend_assets_dir": "./scene_assets/living_room",
    "num_obstacles": (10, 20),
    "num_policy": "random",
    "radius_range": (0.1, 0.5),
    "canvas_x_range": (-1.5, 1.5),
    "canvas_z_range": (0, 3),
    "sphere_probability": 0.3,
}
