WORK_DIR = "/data/zyw/workshop/attempt/work_dirs/manipulation_data"

NUM_DATA = 100
RANDOM_GRAB_DIR = f"{WORK_DIR}/random_grab"
TARGET_DIR = f"{WORK_DIR}/target"
OBSTACLE_DIR = f"{WORK_DIR}/obstacle"
VISUAL_DIR = f"{WORK_DIR}/visual"
FULL_DIR = f"{WORK_DIR}/full"
RELEASE_DIR = f"{WORK_DIR}/release"

random_grab_config = {
    "env_config_path": "./ssim/configs/random_grab.yaml",
    "model_path": "rl_models/train_[-0.3, 0.3, -0.3, 0.3, 0.4, 0.5]",
    "object_range": [-0.3, 0.3, -0.3, 0.3, 0.4, 0.5],
    "target_range": [-0.3, 0.3, -0.3, 0.3, 0.4, 0.5],
    "object_eps": 0.05,
    "target_eps": 0.02,
    "visualize": False,
}

target_config = {
    "radius_range": (0.05, 0.2),
}
re_go_config = {
    "visualize": True,
}
obstacle_config = {
    "num_near_obstacles": 5,
    "num_policy": "fix",
    "radius_range": (0.05, 0.2),
    "x_range": (-0.6, 0.6),
    "y_range": (-0.6, 0.6),
    "z_range": (0.0, 0.5),
}

full_config = {
    "inplace_object_name": False,
}

meshes_config = {
    "base_assets_dir": "./assets",
    "blend_assets_dir": "./scene_assets/living_room",
    "sphere_probability": 0.3,
}
visualize_config = {
    "visualize_2d": True,
    "width": 480,
    "height": 360,
}

release_config = {}
