from tqdm import tqdm

from ssim.arguments import (
    RodArguments,
    SimulatorArguments,
    SphereArguments,
    SuperArgumentParser,
)
from ssim.envs import PushBallEnvironment


def run_simulation(env: PushBallEnvironment):

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

    config_path = "/data/zyw/workshop/attempt/ssim/configs/push_ball.yaml"
    parser = SuperArgumentParser(
        (SphereArguments, RodArguments, SimulatorArguments),
        prefix=("sphere", "rod", "simulator"))
    sphere_config, rod_config, sim_config = parser.parse_yaml_file(config_path)

    env = PushBallEnvironment(rod_config, sphere_config, sim_config)

    env.setup()
    success = run_simulation(env)

    # env.create_3d_animation(save_path=".mp4", fps=sim_config.rendering_fps)
    env.visualize_2d(save_path="2d.mp4")

    return success


if __name__ == "__main__":
    main()
