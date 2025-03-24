from tqdm import tqdm

from ssim.arguments import (RodArguments, SimulatorArguments, SphereArguments,
                            SuperArgumentParser)
from ssim.envs import GrabBallEnvironment, PushBallEnvironment, GrabBallArguments
from ssim.utils import is_contact
import yaml


def run_simulation(env: GrabBallEnvironment) -> bool:

    # Get update interval from simulator configuration
    update_interval = env.sim_config.update_interval

    # Run simulation for a number of steps
    total_steps = env.total_steps
    print(
        f"Starting simulation with {total_steps} total steps and update interval {update_interval}..."
    )

    # Use tqdm to create a progress bar
    progress_steps = range(0, total_steps, update_interval)
    with tqdm(total=total_steps, desc="Simulation Progress") as pbar:
        for _ in progress_steps:
            if not any(env.action_flags):
                for idx, object_ in enumerate(env.objects):
                    if is_contact(object_, env.shearable_rod) and idx == 0:
                        env.action_flags[idx] = True
                        env.uniform_force[-1] = -1
            env.step()
            pbar.update(1)

    return True


def main():

    config_path = "/data/zyw/workshop/attempt/ssim/configs/rod_objects.yaml"
    configs = GrabBallArguments.from_yaml(config_path)

    env = GrabBallEnvironment(configs)

    env.setup()
    success = run_simulation(env)

    env.visualize_2d(video_name="2d.mp4", fps=env.rendering_fps)
    env.visualize_3d(video_name="3d.mp4", fps=env.rendering_fps)
    env.export_callbacks("grab_ball_callbacks.pkl")
    return success


if __name__ == "__main__":
    main()
