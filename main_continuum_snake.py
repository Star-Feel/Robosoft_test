import os
import bpy

from tqdm import tqdm
from ssim.envs import ContinuumSnakeEnvironment, ContinuumSnakeArguments


def run_simulation(env: ContinuumSnakeEnvironment) -> bool:

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
            env.step()
            pbar.update(1)

    return True


def main():

    config_path = "./ssim/configs/continuum_snake.yaml"
    os.chdir('.')
    configs = ContinuumSnakeArguments.from_yaml(config_path)

    env = ContinuumSnakeEnvironment(configs)

    env.setup()
    success = run_simulation(env)
    dir_name = 'continuum_snake'
    # env.visualize_2d(video_name="2d.mp4", fps=env.rendering_fps)
    # env.visualize_3d(video_name="3d.mp4", fps=env.rendering_fps)
    # env.export_callbacks("grab_ball_callbacks.pkl")
    # env.visualize_3d_povray(video_name=f'povray_{dir_name}',
    #                         output_images_dir=f'./work_dirs/povray_{dir_name}',
    #                         fps=20)
    env.visualize_3d_blender(video_name=f'povray_{dir_name}',
                        output_images_dir=f'./work_dirs/povray_{dir_name}',
                        fps=20)
    return success


if __name__ == "__main__":
    main()
