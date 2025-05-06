import os

from tqdm import tqdm
from ssim.components import ChangeableMuscleTorques
from ssim.envs import NavigationSnakeEnvironment, NavigationSnakeArguments


def run_simulation(env: NavigationSnakeEnvironment) -> bool:

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
    os.chdir('.')
    configs = NavigationSnakeArguments.from_yaml(config_path)

    env = NavigationSnakeEnvironment(configs)

    env.setup()
    env.set_target([0, 0.0, 3.5])
    success = run_simulation(env)
    dir_name = 'navigatoin_snake'
    env.visualize_2d(video_name="2d.mp4", fps=env.rendering_fps)
    # env.visualize_3d(video_name="3d.mp4", fps=env.rendering_fps)
    # env.export_callbacks("grab_ball_callbacks.pkl")
    env.visualize_3d_povray(
        video_name=
        f'/data/zyw/workshop/attempt/work_dirs/povray_{dir_name}.mp4',
        output_images_dir=f'./work_dirs/povray_{dir_name}',
        fps=env.rendering_fps)
    return success


if __name__ == "__main__":
    main()
