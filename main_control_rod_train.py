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

    work_dir = "/data/wjs/SoftRoboticaSimulator/work_dir/train_[-0.3, 0.3, -0.3, 0.3, 0.4, 0.5]_dampen10"
    os.makedirs(work_dir, exist_ok=True)
    env = Monitor(env, work_dir)
    log_dir = "/data/wjs/SoftRoboticaSimulator/log_dirs/log_dampen10_new_env_" + identifer + "/"
    os.makedirs(log_dir, exist_ok=True)

    model = SAC(env=env, verbose=1, seed=seed, **items)
    model.set_env(env)

    # Add progress bar
    class ProgressBarCallback:

        def __init__(self, total_timesteps, save_step=1e4):
            self.tracker = 0
            self.save_step = save_step
            self.pbar = tqdm(total=total_timesteps, desc="Training Progress")

        def __call__(self, locals_, globals_):
            self.pbar.update(1)
            self.tracker += 1
            if self.tracker % self.save_step == 0:
                model = locals_["self"]
                data = {
                    "gamma": model.gamma,
                    "verbose": model.verbose,
                    "observation_space": model.observation_space,
                    "action_space": model.action_space,
                    "policy": model.policy,
                    "n_envs": model.n_envs,
                    "n_cpu_tf_sess": model.n_cpu_tf_sess,
                    "seed": model.seed,
                    "action_noise": model.action_noise,
                    "random_exploration": model.random_exploration,
                    "_vectorize_action": model._vectorize_action,
                    "policy_kwargs": model.policy_kwargs
                }
                params = model.get_parameters()
                with open(
                        os.path.join(work_dir, "data_step_" +
                                     str(self.tracker) + ".pth"), "wb") as f:
                    pickle.dump(data, f)
                with open(
                        os.path.join(
                            work_dir,
                            "params_step_" + str(self.tracker) + ".pth"),
                        "wb") as f:
                    pickle.dump(params, f)

            return True

        def close(self):
            self.pbar.close()

    callback = ProgressBarCallback(total_timesteps)

    model.learn(total_timesteps=int(total_timesteps), callback=callback)

    # library helper
    plot_results(
        [work_dir],
        int(total_timesteps),
        results_plotter.X_TIMESTEPS,
        "SAC" + "_" + identifer,
    )
    plt.savefig(os.path.join(work_dir,
                             "convergence_plot" + identifer + ".png"))
    model.save(os.path.join(work_dir, "policy-" + identifer))


if __name__ == "__main__":
    random.seed(42)
    main(42)
