import os
import pickle
import random

import numpy as np
from stable_baselines import SAC
from tqdm import tqdm

from ssim.envs import ControllableGrabArguments, ControllableGrabEnvironment


def main():

    # config_path = "/data/zyw/workshop/attempt/ssim/configs/controllable_rod_objects.yaml"
    config_path = "/data/wjs/SoftRoboticaSimulator/ssim/configs/controllable_rod_objects.yaml"
    work_dir = "/data/zyw/workshop/attempt/work_dirs"
    os.chdir(work_dir)
    configs = ControllableGrabArguments.from_yaml(config_path)

    env = ControllableGrabEnvironment(configs)

    with open(
        "/data/zyw/workshop/Elastica-RL-control-fix-improve2/work_dir/train_origin_xyz_Q_onepoint_norotate_mycond/log_SAC_3d-tracking_id-2000000_0/data.pkl",
        "rb"
    ) as fp:
        data = pickle.load(fp)
    with open(
        "/data/zyw/workshop/Elastica-RL-control-fix-improve2/work_dir/train_origin_xyz_Q_onepoint_norotate_mycond/log_SAC_3d-tracking_id-2000000_0/params.pkl",
        "rb"
    ) as fp:
        param = pickle.load(fp)
    model = SAC(policy=data["policy"], env=None, _init_setup_model=False)  # pytype: disable=not-instantiable
    model.__dict__.update(data)
    model.set_env(None)
    model.setup_model()
    model.load_parameters(param)

    env.setup()

    update_interval = env.sim_config.update_interval

    # Run simulation for a number of steps
    total_steps = env.total_steps
    print(
        f"Starting simulation with {total_steps} total steps and update interval {update_interval}..."
    )
    # Use tqdm to create a progress bar

    progress_steps = range(0, total_steps, update_interval)
    global_step = 0
    with tqdm(total=len(progress_steps), desc="Simulation Progress") as pbar:
        obs = env.set_target_point(
            np.array([
                0.20741026108987037, -0.1671957213830692, 0.16156895806419683
            ]),
            np.array([0, np.pi / 6, 0]),
        )
        for _ in progress_steps:
            global_step += 1
            action, _states = model.predict(obs)
            obs = env.step(action)
            pbar.update(1)
            if env.is_achieve():
                break
        print("Arrive at object")

        if not any(env.action_flags):
            for idx, _ in enumerate(env.objects):
                if env.attach_flags[idx] and idx == 0:
                    env.action_flags[idx] = True
        obs = env.set_target_point(
            np.array([0.3, 0.0, 0.01]),
            np.array([0, np.pi / 6, 0]),
        )

        # for _ in progress_steps:
        #     global_step += 1
        #     action, _states = model.predict(obs)
        #     # action = np.random.uniform(-1, 1, 18)
        #     obs = env.step(action)
        #     pbar.update(1)
        #     if env.is_achieve(0.17):
        #         break
        # print("Arrive at target")

        for i in range(len(env.action_flags)):
            env.action_flags[i] = False

        while True:
            global_step += 1
            action = np.zeros(18)
            obs = env.step(action)
            pbar.update(1)
            if global_step >= len(progress_steps):
                break

    env.visualize_3d(
        video_name="3d.mp4",
        fps=env.rendering_fps,
        xlim=(-0.5, 0.5),
        ylim=(-0.5, 0.5),
        zlim=(0, 1)
    )


if __name__ == "__main__":
    random.seed(42)
    main()
