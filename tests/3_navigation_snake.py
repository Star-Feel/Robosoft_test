import os

from tqdm import tqdm
from ssim.components import ChangeableMuscleTorques
from ssim.envs import (
    NavigationSnakeActionEnvironment, NavigationSnakeArguments
)


def run_simulation(env: NavigationSnakeActionEnvironment) -> bool:

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
    with tqdm(total=total_steps, desc="Simulation Progress") as pbar:
        for i in progress_steps:
            if i == 100000:
                print("turn left")
                env.turn[0] = ChangeableMuscleTorques.LEFT
            if i == 200000:
                env.turn[0] = ChangeableMuscleTorques.DIRECT
            if i == 250000:
                env.turn[0] = ChangeableMuscleTorques.RIGHT
            if i == 300000:
                env.turn[0] = ChangeableMuscleTorques.DIRECT
            if i == 500000:
                print("turn right")
                env.turn[0] = ChangeableMuscleTorques.RIGHT
            if i == 600000:
                env.turn[0] = ChangeableMuscleTorques.DIRECT

            env.step()
            pbar.update(1)
            if env.reach(0.05):
                break

    return True


def main():

    config_path = "./ssim/configs/navigation_snake.yaml"
    work_dir = "./work_dirs/3_navigation_snake"
    os.makedirs(work_dir, exist_ok=True)
    configs = NavigationSnakeArguments.from_yaml(config_path)

    env = NavigationSnakeActionEnvironment(configs)

    env.setup()
    env.set_target([0, 0.0, 3.5])
    success = run_simulation(env)
    env.visualize_2d(
        video_name=os.path.join(work_dir, "2d.mp4"), fps=env.rendering_fps
    )
    # env.visualize_3d(video_name="3d.mp4", fps=env.rendering_fps)
    # env.export_callbacks("grab_ball_callbacks.pkl")

    return success


if __name__ == "__main__":
    main()
