import os.path as osp
import pickle

import numpy as np
from tqdm import tqdm

from ssim.envs import (
    NavigationSnakeArguments,
    NavigationSnakeTorqueEnvironment,
)
from ssim.utils import load_json


def main():
    # data_path = "/data/zyw/workshop/attempt/work_dirs/navigation_data/full/0/info.json"
    # work_dir = "/data/zyw/workshop/attempt/work_dirs/navigation_demo"
    # os.chdir("/data/wjs/wrp/SoftRoboticaSimulator")
    data_path = "work_dirs/navigation_data/full/5/info.json"
    work_dir = "work_dirs/navigation_demo"
    info = load_json(data_path)

    config = NavigationSnakeArguments.from_yaml(info["config"])
    env = NavigationSnakeTorqueEnvironment(config)
    env.setup()
    # env.set_target(info["target_id"])

    with open(info["state_action"], "rb") as f:
        state_action = pickle.load(f)
    actions = state_action["torque"]

    try:
        for i, action in tqdm(
            enumerate(actions), total=len(actions), desc="Simulation"
        ):
            env.step(action)
            # if i > 1e4:
            #     break
            # if np.any(env.attach_flags):
            #     print(env.attach_flags)
            #     break
    except Exception as e:
        print(e)
        print("Simulation interrupted.")
        # if env.reach():
        #     break

    env.visualize_2d(
        osp.join(work_dir, "demo.mp4"),
        xlim=(-1,3),
        ylim=(-2, 1),
        target_last=True
    )

    # env.visualize_3d_povray(video_name=osp.join(work_dir, "demo_povray"),
    #                         output_images_dir=osp.join(work_dir, "povray"),
    #                         fps=env.rendering_fps)

    # env.visualize_3d_blender(
    #     video_name='test',
    #     output_images_dir=osp.join(work_dir,'test/povray_test'),
    #     fps=15,
    #     width=480,
    #     height=360,
    # )


if __name__ == "__main__":
    np.random.seed(0)
    main()
