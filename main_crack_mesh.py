
import os

from tqdm import tqdm

from ssim.components.surface.mesh_surface import MeshSurface
from ssim.envs import MeshDemoArguments, MeshDemoEnvironment
from ssim.utils import is_contact


def run_simulation(env: MeshDemoEnvironment) -> bool:

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
        counter = 0
        for _ in progress_steps:
            if not any(env.action_flags):
                for idx, object_ in enumerate(env.objects):
                    if isinstance(object_, MeshSurface):
                        continue
                    if is_contact(
                            object_,
                            env.shearable_rod) and idx == 0 and counter == 0:
                        env.action_flags[idx] = True
                        env.uniform_force[-1] = -1
            else:
                counter += 1
            if counter >= 10000:
                for i in range(len(env.action_flags)):
                    env.action_flags[i] = False
                    env.uniform_force[-1] = 0
            env.step()
            pbar.update(1)

    return True


def main():

    config_path = "/data/zyw/workshop/attempt/ssim/configs/rod_mesh.yaml"
    work_dir = "/data/zyw/workshop/attempt/work_dirs"
    os.chdir(work_dir)
    configs = MeshDemoArguments.from_yaml(config_path)

    env = MeshDemoEnvironment(configs)

    env.setup()
    success = run_simulation(env)

    env.visualize_2d(video_name="2d.mp4", fps=env.rendering_fps)
    # env.visualize_3d(video_name="3d.mp4", fps=env.rendering_fps)
    env.export_callbacks("grab_mesh_callbacks.pkl")
    env.visualize_3d_povray(video_name='povray',
                            output_images_dir='./work_dirs/povray',
                            fps=20)
    return success


if __name__ == "__main__":

    main()
