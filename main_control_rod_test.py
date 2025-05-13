import os
import pickle
import random
from tqdm import tqdm

from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy as MlpPolicy_SAC

from ssim.envs import SoftTargetControlArguments, SoftTargetControlEnvironment
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import ts2xy, plot_results
from stable_baselines.bench.monitor import Monitor, load_results
import numpy as np
from stable_baselines import results_plotter

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def main(seed: int = 42):

    config_path = "/data/wjs/SoftRoboticaSimulator/ssim/configs/rod_target_train.yaml"
    
    configs = SoftTargetControlArguments.from_yaml(config_path)
    env = SoftTargetControlEnvironment(configs)
    
    timesteps_per_batch = 2000000
    total_timesteps = 10000000

    items = {
        "policy": MlpPolicy_SAC,
        "buffer_size": int(timesteps_per_batch),
    }

    name = "SAC_3d-tracking_id"
    identifer = name + "-" + str(timesteps_per_batch) + "_" + str(seed)

    with open(
            "/data/wjs/SoftRoboticaSimulator/work_dir/train_[-0.6, 0.6, -0.6, 0.6, 0.0, 0.3]_dampen5/data_step_10000000.pth",
            "rb") as fp:
        data = pickle.load(fp)
    with open(
            "/data/wjs/SoftRoboticaSimulator/work_dir/train_[-0.6, 0.6, -0.6, 0.6, 0.0, 0.3]_dampen5/params_step_10000000.pth",
            "rb") as fp:
        param = pickle.load(fp)

    model = SAC(env=env, verbose=1, seed=seed, **items)
    model.__dict__.update(data)
    model.load_parameters(param)
    model.setup_model()
    model.buffer_size = 2000000
    model.set_env(env)

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

        for _ in progress_steps:
            global_step += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            score += rewards            
            pbar.update(1)
            # if env.is_achieve():
            #     break
        print('Rod end point arrives at object.')

        # if not any(env.action_flags):
        #     for idx, object_ in enumerate(env.objects):
        #         if env.attach_flags[idx] and idx == 0:
        #             env.action_flags[idx] = True    

    print("Final Score:", score)
    env.visualize_3d(
        video_name="/data/wjs/SoftRoboticaSimulator/test_result/dampen5_10000000_test.mp4", 
        fps=env.rendering_fps,
        xlim=(-0.6, 0.6),
        ylim=(-0.6, 0.6),
        zlim=(0, 1),
    )

if __name__ == "__main__":
    random.seed(42)
    main(42)
