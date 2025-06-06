import os
import os.path as osp
import pickle
from dataclasses import dataclass

import numpy as np
import yaml
from stable_baselines import SAC
from tqdm import tqdm

from configs import NUM_DATA, RANDOM_GRAB_DIR, random_grab_config
from ssim.envs import ControllableGrabArguments, ControllableGrabEnvironment
from ssim.visualize.visualizer import plot_contour_3d


def export_config(config, output_dir: str):

    with open(osp.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)


@dataclass
class RandomGrabConfig:
    env_config_path: str
    model_path: str
    object_range: list
    target_range: list
    object_eps: float
    target_eps: float
    visualize: bool


def main():
    script_config = RandomGrabConfig(**random_grab_config)

    # Setting RL model
    with open(osp.join(script_config.model_path, "data.pkl"), "rb") as fp:
        data = pickle.load(fp)
    with open(osp.join(script_config.model_path, "params.pkl"), "rb") as fp:
        param = pickle.load(fp)

    model = SAC(policy=data["policy"], env=None, _init_setup_model=False)
    model.__dict__.update(data)
    model.set_env(None)
    model.setup_model()
    model.load_parameters(param)
    model.buffer_size = 2000000

    object_range = script_config.object_range
    target_range = script_config.target_range

    for i in range(NUM_DATA):
        print(f"Generating data {i + 1}/{NUM_DATA}...")
        local_output_dir = osp.join(RANDOM_GRAB_DIR, f"{i}")
        os.makedirs(local_output_dir, exist_ok=True)

        configs = ControllableGrabArguments.from_yaml(
            script_config.env_config_path
        )
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
            f"Starting simulation with {total_steps} total steps and "
            f"update interval {update_interval}..."
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

            obs = env.set_target_point(
                # 指定位置
                configs.objects[0].center,
                np.array([0, np.random.uniform(np.pi / 6, np.pi / 3), 0]),
            )

            for i in progress_steps:
                process_steps.append(global_step)

                rod_position, rod_velocity = env.get_rod_state()
                rod_position_records.append(rod_position)
                rod_velocity_records.append(rod_velocity)

                action, _ = model.predict(obs)
                obs, _, _, _ = env.step(action)

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
                if env.is_rod_achieve_target(script_config.object_eps):
                    break
            time_shots.append(global_step)
            print('Rod end point arrives at object.')

            if not any(env.action_flags):
                for idx, _ in enumerate(env.objects):
                    if env.attach_flags[idx] and idx == 0:
                        env.action_flags[idx] = True

            # setting target position
            obs = env.set_target_point(
                configs.objects[1].center,
                np.array([0, np.random.uniform(np.pi / 6, np.pi / 3), 0]),
            )

            for _ in progress_steps:
                process_steps.append(global_step)

                rod_position, rod_velocity = env.get_rod_state()
                rod_position_records.append(rod_position)
                rod_velocity_records.append(rod_velocity)

                action, _ = model.predict(obs)
                obs, _, _, _ = env.step(action)

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
                if env.is_rod_achieve_target(script_config.target_eps):
                    break
            time_shots.append(global_step)
            print('Rod end point arrives at target.')

        with open(script_config.env_config_path, "r") as f:
            env_config = yaml.safe_load(f)
        env_config['objects'] = []

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

        plot_contour_3d(
            positions=np.array(env.rod_callback["position"]
                               ).transpose(0, 2, 1),
            save_path=osp.join(local_output_dir, "contour.png")
        )
        if script_config.visualize:
            env.visualize_3d(
                video_name=osp.join(local_output_dir, "3d.mp4"),
                fps=env.rendering_fps,
                xlim=(-0.6, 0.6),
                ylim=(-0.6, 0.6),
                zlim=(0, 1),
            )


if __name__ == "__main__":
    main()
