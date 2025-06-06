import os
import os.path as osp
import pickle
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from configs import (
    FULL_DIR,
    NUM_DATA,
    RANDOM_GRAB_DIR,
    TARGET_DIR,
    re_go_config,
)
from ssim.envs import ControllableGrabArguments, ControllableGrabEnvironment
from ssim.utils import load_yaml


@dataclass
class ReGoConfig:
    visualize: bool


def main():
    script_config = ReGoConfig(**re_go_config)
    for i in range(NUM_DATA):
        print(f"processing {i}/{NUM_DATA}")
        local_random_dir = osp.join(RANDOM_GRAB_DIR, f"{i}")
        local_target_object_dir = osp.join(TARGET_DIR, f"{i}")
        local_target_dir = osp.join(FULL_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        object_id = 0
        target_id = 1
        # Setting environment
        config_path = osp.join(local_target_object_dir, "config.yaml")
        state_action_path = osp.join(local_random_dir, "state_action.pkl")
        with open(state_action_path, "rb") as f:
            state_action = pickle.load(f)
        config = load_yaml(config_path)
        target_config = config["objects"][target_id]
        configs = ControllableGrabArguments.from_yaml(config_path)
        env = ControllableGrabEnvironment(configs)
        env.setup()
        env.set_target_object(object_id)
        env.set_target_point(
            np.array(target_config["center"]),
            np.array([0, 0, 0]),
        )

        update_interval = env.sim_config.update_interval
        total_steps = env.total_steps
        print(
            f"Starting simulation with {total_steps} total steps and "
            f"update interval {update_interval}..."
        )
        progress_steps = range(0, total_steps, update_interval)
        global_step = 0
        with tqdm(
            total=2 * len(progress_steps), desc="Simulation Progress"
        ) as pbar:

            pick_step = -1
            place_step = -1
            for i in progress_steps:
                if global_step >= state_action["normal_torque"].shape[0]:
                    break

                normal_torque = state_action["normal_torque"][global_step]
                binormal_torque = state_action["binormal_torque"][global_step]
                twist_torque = state_action["twist_torque"][global_step]
                action = np.concatenate(
                    (normal_torque, binormal_torque, twist_torque), axis=0
                )
                env.step(action)

                pbar.update(1)
                global_step += 1
                if not any(env.action_flags
                           ) and env.is_rod_achieve_object_by_feedback():
                    print('Rod end point arrives at object.')
                    pick_step = global_step
                    env.action_flags[object_id] = True

                if (
                    any(env.action_flags)
                    and env.is_object_achieve_target(0.001)
                ):
                    print('Rod end point arrives at target.')
                    place_step = global_step
                    break
        del state_action["time_shots"]
        new_state_action = {
            k: v[:global_step + 1]
            for k, v in state_action.items()
        }
        new_state_action["action_steps"] = {
            "pick": pick_step,
            "place": place_step if place_step != -1 else global_step,
        }
        with open(osp.join(local_target_dir, "state_action.pkl"), "wb") as f:
            pickle.dump(new_state_action, f)
        if script_config.visualize:
            env.visualize_3d(
                video_name=osp.join(local_target_dir, '3d.mp4'),
                fps=env.rendering_fps,
                xlim=(-0.6, 0.6),
                ylim=(-0.6, 0.6),
                zlim=(0, 1),
            )


if __name__ == "__main__":
    main()
