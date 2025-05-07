import os
import sys
import numpy as np
import os.path as osp
import pickle
import json

sys.path.append("/data/zyw/workshop/attempt")

import yaml

from ssim.utils import load_json

sys.path.append("/data/zyw/workshop/attempt")
from tqdm import tqdm
from ssim.components import ChangeableMuscleTorques
from ssim.envs import NavigationSnakeTorqueEnvironment, NavigationSnakeArguments
from ssim.visualize.visualizer import plot_contour

N = 100


def main():
    source_dir = "/data/zyw/workshop/attempt/work_dirs/navigation_data/full/0/info.json"
    work_dir = "/data/zyw/workshop/attempt/work_dirs/navigation_demo"
    info = load_json(source_dir)

    config = NavigationSnakeArguments.from_yaml(info["config"])
    env = NavigationSnakeTorqueEnvironment(config)
    env.setup()
    env.set_target(info["target_id"])

    with open(info["state_action"], "rb") as f:
        state_action = pickle.load(f)
    actions = state_action["torque"]
    for action in tqdm(actions):
        env.step(action)
        if env.reach():
            break

    env.visualize_2d(osp.join(work_dir, "demo.mp4"), equal_aspect=True)


if __name__ == "__main__":
    np.random.seed(0)
    main()
