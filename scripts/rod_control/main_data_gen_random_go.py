import os
import os.path as osp
import pickle

import numpy as np
import yaml
from stable_baselines import SAC
from tqdm import tqdm

from ssim.envs import ControllableGrabArguments, ControllableGrabEnvironment
from ssim.visualize.visualizer import plot_contour

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

Gen_data_Num = 100


def export_state_action(env: ControllableGrabEnvironment, output_dir: str):
    torque_normal_callback = env.torque_profile_list_for_muscle_in_normal_dir
    torque_binormal_callback = env.torque_profile_list_for_muscle_in_binormal_dir
    torque_twist_callback = env.torque_profile_list_for_muscle_in_twist_dir
    rod_callback = env.rod_callback
    state_action = {
        "rod_time": rod_callback["time"],
        "rod_position": rod_callback["position"],
        "rod_directors": rod_callback["directors"],
        "rod_velocity": rod_callback["velocity"],
        "rod_torque_normal_time": torque_normal_callback['time'],
        "rod_torque_normal": torque_normal_callback['torque'],
        "rod_torque_normal_mag": torque_normal_callback['torque_mag'],
        "rod_torque_binormal_time": torque_binormal_callback['time'],
        "rod_torque_binormal": torque_binormal_callback['torque'],
        "rod_torque_binormal_mag": torque_binormal_callback['torque_mag'],
        "rod_torque_twist_time": torque_twist_callback['time'],
        "rod_torque_twist": torque_twist_callback['torque'],
        "rod_torque_twist_mag": torque_twist_callback['torque_mag'],
    }
    state_action_path = osp.join(output_dir, "state_action.pkl")
    with open(state_action_path, "wb") as f:
        pickle.dump(state_action, f)


def export_config(config, output_dir: str):

    with open(osp.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)


object_range = [-0.3, 0.3, -0.3, 0.3, 0.4, 0.5]
target_range = [-0.3, 0.3, -0.3, 0.3, 0.4, 0.5]
rl_model_data_path = "rl_models/train_[-0.3, 0.3, -0.3, 0.3, 0.4, 0.5]/data.pkl"
rl_model_param_path = "rl_models/train_[-0.3, 0.3, -0.3, 0.3, 0.4, 0.5]/params.pkl"


def main():

    # Generate data setting
    output_dir = "./work_dirs/rod_control_data/random_go"
    os.makedirs(output_dir, exist_ok=True)

    # Setting environment
    config_path = "./ssim/configs/random_go.yaml"

    configs = ControllableGrabArguments.from_yaml(config_path)
    env = ControllableGrabEnvironment(configs)

    # Setting RL model
    with open(rl_model_data_path, "rb") as fp:
        data = pickle.load(fp)
    with open(rl_model_param_path, "rb") as fp:
        param = pickle.load(fp)

    model = SAC(policy=data["policy"], env=None, _init_setup_model=False)  # pytype: disable=not-instantiable
    model.__dict__.update(data)
    model.set_env(None)
    model.setup_model()
    model.load_parameters(param)
    model.buffer_size = 2000000

    for i in range(Gen_data_Num):
        print(f"Generating data {i + 1}/{Gen_data_Num}...")
        local_output_dir = osp.join(output_dir, f"{i}")
        os.makedirs(local_output_dir, exist_ok=True)

        t_x = np.random.uniform(object_range[0], object_range[1])
        t_y = np.random.uniform(object_range[2], object_range[3])
        t_z = np.random.uniform(object_range[4], object_range[5])
        print("Start position:", t_x, t_y, t_z)
        # start_postition = np.array([t_x, t_y, t_z])
        configs.objects[0].center = np.array([t_x, t_y, t_z])

        t_x = np.random.uniform(target_range[0], target_range[1])
        t_y = np.random.uniform(target_range[2], target_range[3])
        t_z = np.random.uniform(target_range[4], target_range[5])
        print("Target position:", t_x, t_y, t_z)
        # target_postition = np.array([t_x, t_y, t_z])
        configs.objects[1].center = np.array([t_x, t_y, t_z])

        env = ControllableGrabEnvironment(configs)
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

        rod_position_records = []
        rod_velocity_records = []
        normal_torque_records = []
        binormal_torque_records = []
        twist_torque_records = []
        time_shots = []
        process_steps = []

        with tqdm(
            total=2 * len(progress_steps), desc="Simulation Progress"
        ) as pbar:

            obs = env.set_target(
                # 指定位置
                configs.objects[0].center,
                np.array([0, np.random.uniform(np.pi / 6, np.pi / 3), 0]),
            )

            for i in progress_steps:
                process_steps.append(global_step)

                rod_position, rod_velocity = env.get_rod_state()
                rod_position_records.append(rod_position)
                rod_velocity_records.append(rod_velocity)

                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)

                normal_torque_records.append(
                    action[:env.number_of_control_points]
                )
                binormal_torque_records.append(
                    action[env.number_of_control_points:2
                           * env.number_of_control_points]
                )
                twist_torque_records.append(
                    action[2 * env.number_of_control_points:]
                )

                pbar.update(1)
                global_step += update_interval
                if env.is_achieve(0.05):
                    break
            time_shots.append(global_step)
            print('Rod end point arrives at object.')

            if not any(env.action_flags):
                for idx, object_ in enumerate(env.objects):
                    if env.attach_flags[idx] and idx == 0:
                        env.action_flags[idx] = True

            # setting target position
            obs = env.set_target(
                configs.objects[1].center,
                np.array([0, np.random.uniform(np.pi / 6, np.pi / 3), 0]),
            )

            for _ in progress_steps:
                process_steps.append(global_step)

                rod_position, rod_velocity = env.get_rod_state()
                rod_position_records.append(rod_position)
                rod_velocity_records.append(rod_velocity)

                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)

                normal_torque_records.append(
                    action[:env.number_of_control_points]
                )
                binormal_torque_records.append(
                    action[env.number_of_control_points:2
                           * env.number_of_control_points]
                )
                twist_torque_records.append(
                    action[2 * env.number_of_control_points:]
                )

                pbar.update(1)
                global_step += update_interval
                if env.is_achieve(0.02):
                    break
            time_shots.append(global_step)
            print('Rod end point arrives at object.')

        # export_state_action(env, local_output_dir)

        with open(config_path, "r") as f:
            env_config = yaml.safe_load(f)
        env_config['objects'] = []
        # env_config['objects'][1]['center'] = configs.objects[1].center.tolist()

        export_config(env_config, local_output_dir)
        state_action = {
            "rod_position": np.array(rod_position_records),
            "rod_velocity": np.array(rod_velocity_records),
            "normal_torque": np.array(normal_torque_records),
            "binormal_torque": np.array(binormal_torque_records),
            "twist_torque": np.array(twist_torque_records),
            "time_shots": np.array(time_shots),
            "process_steps": np.array(process_steps),
        }
        with open(osp.join(local_output_dir, "state_action.pkl"), "wb") as f:
            pickle.dump(state_action, f)

        # env.visualize_2d(video_name=osp.join(local_output_dir, "2d.mp4"), fps=env.rendering_fps)
        env.visualize_3d(
            video_name=osp.join(local_output_dir, "3d.mp4"),
            fps=env.rendering_fps,
            xlim=(-0.6, 0.6),
            ylim=(-0.6, 0.6),
            zlim=(0, 1),
        )

        # contour_xy
        plot_contour(
            positions=np.array(env.rod_callback["position"]
                               ).transpose(0, 2, 1)[..., [0, 2]],
            save_path=osp.join(local_output_dir, "contour.png")
        )

        # contour_xz
        # plot_contour(
        #     positions=np.array(env.rod_callback["position"]).transpose(0, 2, 1)[..., [0, 1]],
        #     save_path=osp.join(local_output_dir, "contour_xz.png")
        # )


if __name__ == "__main__":
    main()
