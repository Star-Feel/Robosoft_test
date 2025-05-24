import os
import os.path as osp
import pickle

import numpy as np
from tqdm import tqdm

from ssim.envs import ControllableGrabArguments, ControllableGrabEnvironment
from ssim.utils import load_json, load_yaml

N = 50
RANDOM_DIR = "./work_dirs/rod_control_data/random_go"
TARGET_OBJECT_DIR = "./work_dirs/rod_control_data/target"
TARGET_DIR = "./work_dirs/rod_control_data/full"


def main():
    for i in range(N):
        print(f"processing {i}/{N}")
        local_random_dir = osp.join(RANDOM_DIR, f"{i}")
        local_target_object_dir = osp.join(TARGET_OBJECT_DIR, f"{i}")
        local_target_dir = osp.join(TARGET_DIR, f"{i}")
        os.makedirs(local_target_dir, exist_ok=True)

        object_id = 0
        target_id = 1
        # Setting environment
        config_path = osp.join(local_target_object_dir, "config.yaml")
        state_action_path = osp.join(local_random_dir, "state_action.pkl")
        with open(state_action_path, "rb") as f:
            state_action = pickle.load(f)
        config = load_yaml(config_path)
        object_config = config["objects"][object_id]
        target_config = config["objects"][target_id]
        configs = ControllableGrabArguments.from_yaml(config_path)
        env = ControllableGrabEnvironment(configs)
        env.setup()

        update_interval = env.sim_config.update_interval
        total_steps = env.total_steps
        print(
            f"Starting simulation with {total_steps} total steps and update interval {update_interval}..."
        )
        progress_steps = range(0, total_steps, update_interval)
        global_step = 0
        with tqdm(
            total=2 * len(progress_steps), desc="Simulation Progress"
        ) as pbar:

            env.set_target(
                # 指定位置
                np.array(object_config["center"]),
                np.array([0, np.random.uniform(np.pi / 6, np.pi / 3), 0]),
            )

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
                if env.is_achieve(object_config["radius"]):
                    print('Rod end point arrives at object.')
                    pick_step = global_step
                    if not any(env.action_flags):
                        env.action_flags[object_id] = True
                    env.set_target(
                        np.array(target_config["center"]),
                        np.array([
                            0, np.random.uniform(np.pi / 6, np.pi / 3), 0
                        ]),
                    )

                if any(env.action_flags
                       ) and env.is_achieve(object_config["radius"]):
                    print('Rod end point arrives at object.')
                    place_step = global_step
                    break
        del state_action["time_shots"]
        new_state_action = {
            k: v[:global_step + 1]
            for k, v in state_action.items()
        }
        new_state_action["action_steps"] = {
            "pick": pick_step,
            "place": place_step,
        }
        with open(osp.join(local_target_dir, "state_action.pkl"), "wb") as f:
            pickle.dump(new_state_action, f)


if __name__ == "__main__":
    main()
