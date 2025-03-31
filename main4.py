import os
from turtle import update

from tqdm import tqdm

from ssim.envs import SoftGrabEnvironment, SoftGrabArguments
from ssim.utils import is_contact
from ssim.components.surface.mesh_surface import MeshSurface
import random


def run_simulation(env: SoftGrabEnvironment) -> bool:

    # Get update interval from simulator configuration
    update_interval = env.sim_config.update_interval

    # Run simulation for a number of steps
    total_steps = env.total_steps
    print(
        f"Starting simulation with {total_steps} total steps and update interval {update_interval}..."
    )

    # Use tqdm to create a progress bar
    progress_steps = range(0, total_steps, update_interval)
    with tqdm(total=total_steps // update_interval,
              desc="Simulation Progress") as pbar:

        for _ in progress_steps:
            if not any(env.action_flags):
                for idx, object_ in enumerate(env.objects):
                    if is_contact(object_, env.shearable_rod):
                        env.action_flags[idx] = True
            force_direction = random.randint(0, 2)
            force_value = random.uniform(-1, 1) * 200
            env.uniform_force[force_direction] = force_value

            env.step(update_interval)
            pbar.update(1)

    return True


def main():

    config_path = "/data/zyw/workshop/attempt/ssim/configs/rod_grab.yaml"
    work_dir = "/data/zyw/workshop/attempt/work_dirs"
    os.chdir(work_dir)
    configs = SoftGrabArguments.from_yaml(config_path)

    env = SoftGrabEnvironment(configs)

    env.setup()
    success = run_simulation(env)

    # env.visualize_2d(video_name="2d.mp4", fps=env.rendering_fps)
    env.visualize_3d(video_name="3d.mp4",
                     fps=env.rendering_fps,
                     xlim=(-4, 4),
                     ylim=(-4, 4),
                     zlim=(-1, 4))
    env.export_callbacks("grab_mesh_callbacks.pkl")
    return success


if __name__ == "__main__":
    random.seed(42)
    main()
