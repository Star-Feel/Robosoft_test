import os
import pickle
import numpy as np

from tqdm import tqdm
from ssim.envs import ControllableGrabArguments, ControllableGrabEnvironment
from ssim.utils import is_contact
from stable_baselines import SAC

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def main():

    config_path = "/data/wjs/SoftRoboticaSimulator/ssim/configs/controllable_rod_objects.yaml"
    work_dir = "/data/wjs/SoftRoboticaSimulator/work_dirs"
    os.chdir(work_dir)

    # Setting environment
    configs = ControllableGrabArguments.from_yaml(config_path)
    print(configs.objects[0].center)
    env = ControllableGrabEnvironment(configs)

    # Setting RL model
    # MLP = MlpPolicy_SAC
    # algo = SAC
    # model = algo.load('/data/zyw/workshop/Elastica-RL-control-fix-improve2/work_dir/train_origin_xyz_Q_onepoint_norotate_mycond/log_SAC_3d-tracking_id-2000000_0/policy-SAC_3d-tracking_id-2000000_0_step_10000000.zip')
    with open(
        "/data/wjs/SoftRoboticaSimulator/work_dir/train_[-0.6, 0.6, -0.6, 0.6, 0.0, 0.3]_dampen5/data_step_10000000.pth",
        "rb"
    ) as fp:
        data = pickle.load(fp)
    with open(
        "/data/wjs/SoftRoboticaSimulator/work_dir/train_[-0.6, 0.6, -0.6, 0.6, 0.0, 0.3]_dampen5/params_step_10000000.pth",
        "rb"
    ) as fp:
        param = pickle.load(fp)

    model = SAC(policy=data["policy"], env=None, _init_setup_model=False)  # pytype: disable=not-instantiable
    model.__dict__.update(data)
    model.set_env(None)
    model.setup_model()
    model.load_parameters(param)
    model.buffer_size = 2000000

    env.setup()

    # Get update interval from simulator configuration
    update_interval = env.sim_config.update_interval

    # Run simulation for a number of steps
    total_steps = env.total_steps
    print(
        f"Starting simulation with {total_steps} total steps and update interval {update_interval}..."
    )

    # Use tqdm to create a progress bar
    progress_steps = range(0, total_steps, update_interval)
    global_step = 0
    score = 0

    with tqdm(total=total_steps, desc="Simulation Progress") as pbar:
        print(configs.objects[0].center)
        obs = env.set_target_point(
            # 指定位置
            # np.array([-0.25766822, -0.01381743, 0.03295922]),
            configs.objects[0].center,
            np.array([0, np.random.uniform(np.pi / 6, np.pi / 3), 0]),
        )

        for _ in progress_steps:
            global_step += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            score += rewards
            pbar.update(1)
            if env.is_achieve():
                break
        print('Rod end point arrives at object.')
        print("Final Score:", score)
        score = 0

        if not any(env.action_flags):
            for idx, object_ in enumerate(env.objects):
                if env.attach_flags[idx] and idx == 0:
                    env.action_flags[idx] = True

        # setting target position
        print(configs.objects[1].center)
        obs = env.set_target_point(
            configs.objects[1].center,
            np.array([0, np.random.uniform(np.pi / 6, np.pi / 3), 0]),
        )

        for _ in progress_steps:
            global_step += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            score += rewards
            pbar.update(1)
            if env.is_achieve():
                break
        print('Rod end point arrives at object.')

        # for i in range(len(env.action_flags)):
        #     env.action_flags[i] = False

        # while True:
        #     global_step += 1
        #     action, _states = model.predict(obs)
        #     obs, rewards, done, info = env.step(action)
        #     score += rewards
        #     pbar.update(1)
        #     if global_step >= len(progress_steps) :
        #         break

        # while True:
        #     global_step += 1
        #     action = np.zeros(18)
        #     obs = env.step(action)
        #     pbar.update(1)
        #     if global_step >= len(progress_steps):
        #         break

    print("Final Score:", score)
    env.visualize_3d(
        video_name="/data/wjs/SoftRoboticaSimulator/test_result/3d_catch_2.mp4",
        fps=env.rendering_fps,
        xlim=(-0.6, 0.6),
        ylim=(-0.6, 0.6),
        zlim=(0, 1),
    )


if __name__ == "__main__":
    main()
