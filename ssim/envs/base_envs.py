__all__ = ["RodObjectsEnvironment"]

import pickle
from abc import ABC, abstractmethod

import elastica as ea
import matplotlib.pyplot as plt
import numpy as np
from elastica import (BaseSystemCollection, CallBacks, Connections,
                      Constraints, Contact, Damping, Forcing, PositionVerlet)
from elastica.timestepper import extend_stepper_interface
from matplotlib import animation
from tqdm import tqdm

from ..arguments import SphereArguments, MeshSurfaceArguments
from ..components import MeshSurface, RigidBodyCallBack, RodCallBack
from ..components.callback import MeshSurfaceCallBack
from ..visualize.visualizer import rod_objects_3d_visualize
from ..visualize.renderer import POVRayRenderer


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing,
                    Damping, Contact, CallBacks):
    """Base simulator class combining Elastica functionalities."""
    pass


class SimulateMixin:

    def __init__(self,
                 *args,
                 final_time: float,
                 time_step: int,
                 update_interval: int = 1,
                 rendering_fps: int = 60,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.simulator = BaseSimulator()

        self.final_time = final_time
        self.time_step = time_step
        self.update_interval = update_interval
        self.rendering_fps = rendering_fps
        self.stateful_stepper = PositionVerlet()
        self.total_steps = int(final_time / time_step)
        self.time_step = np.float64(float(final_time) / self.total_steps)
        self.time_tracker = np.float64(0.0)

    def _do_step(self):
        self.time_tracker = self.do_step(
            self.stateful_stepper,
            self.stages_and_updates,
            self.simulator,
            self.time_tracker,
            self.time_step,
        )

    def _finalize(self):
        # Finalize the simulator
        self.simulator.finalize()

        # Prepare for time stepping
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.stateful_stepper, self.simulator)


class RodObjectsMixin:

    simulator: BaseSimulator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shearable_rod = None

        self.objects = []
        self.action_flags = []
        self.attach_flags = []
        self.attachable_flags = []
        self.object2id = {}
        self.object_callbacks = []

        self.rod_callback = ea.defaultdict(list)
        self.objects_callback = []

    def add_shearable_rod(
        self,
        n_elem: int,
        start: np.ndarray,
        direction: np.ndarray,
        normal: np.ndarray,
        base_length: float,
        base_radius: float,
        density: float,
        youngs_modulus: float,
        shear_modulus: float,
    ):
        self.shearable_rod = ea.CosseratRod.straight_rod(
            n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
        )
        self.simulator.append(self.shearable_rod)

    def add_sphere(
        self,
        center: np.ndarray,
        radius: float,
        density: float,
    ):
        """
        Add a sphere to the environment.

        Args:
            center (np.ndarray): Center of the sphere.
            radius (float): Radius of the sphere.
        """
        sphere = ea.Sphere(center, radius, density=density)
        self.simulator.append(sphere)

        self.objects.append(sphere)
        self.object2id[sphere] = len(self.objects) - 1
        self.action_flags.append(False)
        self.attach_flags.append(False)
        self.attachable_flags.append(False)
        self.object_callbacks.append(ea.defaultdict(list))

        # if self.shearable_rod is not None:
        #     self.simulator.constrain(sphere).using(
        #         PinJoint,
        #         other=self.shearable_rod,
        #         index=-1,
        #         flag=self.action_flags,
        #         flag_id=self.object2id[sphere],
        #     )

    def add_cylinder(self):
        pass

    def add_mesh_surface(self,
                         mesh_path: str,
                         center: np.ndarray = np.array([0., 0., 0.]),
                         scale: np.ndarray = np.array([1., 1., 1.]),
                         rotate: np.ndarray = np.array([0., 0., 0.])):
        mesh = MeshSurface(mesh_path)
        mesh.scale(scale)
        mesh.rotate(np.array([1, 0, 0]), rotate[0])
        mesh.rotate(np.array([0, 1, 0]), rotate[1])
        mesh.rotate(np.array([0, 0, 1]), rotate[2])
        mesh.translate(center)
        self.simulator.append(mesh)

        self.objects.append(mesh)
        self.object2id[mesh] = len(self.objects) - 1
        self.action_flags.append(False)
        self.attach_flags.append(False)
        self.attachable_flags.append(False)
        self.object_callbacks.append(ea.defaultdict(list))

    def _add_data_collection_callbacks(self, step_skip: int):
        if self.shearable_rod is not None:
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                RodCallBack,
                step_skip=step_skip,
                callback_params=self.rod_callback)

        for object_ in self.objects:
            if isinstance(object_, ea.RigidBodyBase):
                self.simulator.collect_diagnostics(object_).using(
                    RigidBodyCallBack,
                    step_skip=step_skip,
                    callback_params=self.object_callbacks[
                        self.object2id[object_]])
            elif isinstance(object_, MeshSurface):
                self.simulator.collect_diagnostics(object_).using(
                    MeshSurfaceCallBack,
                    step_skip=step_skip,
                    callback_params=self.object_callbacks[
                        self.object2id[object_]])

    def export_callbacks(self, filename):
        """
        Export the collected callback data to a file.

        Args:
            filename (str): The name of the file to save the callback data.
        """

        callback_data = {
            "deciption":
            "rod is the softrobot, whose callback is a dict of positions,"
            "velocities... \nobjects is all spheres, whose callback is a list"
            "of each object's callback",
            "rod_callback":
            self.rod_callback,
            "object_callbacks":
            self.object_callbacks,
        }

        with open(filename, 'wb') as f:
            pickle.dump(callback_data, f)

    def visualize_2d(
        self,
        video_name="video.mp4",
        fps=15,
        xlim=None,
        ylim=None,
        skip=1,
        equal_aspect=False,
        target_last=False,
    ):

        positions_over_time = np.array(self.rod_callback["position"])
        positions_over_time = positions_over_time[::skip]
        object_positions = [
            np.array(params["position"][::skip])
            for params in self.object_callbacks
        ]

        all_positions = np.concatenate([
            positions_over_time.transpose(0, 2, 1).reshape(-1, 3),
            *[pos.reshape(-1, 3) for pos in object_positions]
        ],
                                       axis=0)
        # Automatically set xlim and ylim if not provided
        if xlim is None:
            x_min = np.min(all_positions[:, 2])
            x_max = np.max(all_positions[:, 2])
            xlim = (x_min, x_max)

        if ylim is None:
            y_min = np.min(all_positions[:, 0])
            y_max = np.max(all_positions[:, 0])
            ylim = (y_min, y_max)

        if equal_aspect:
            max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
            xlim = (xlim[0], xlim[0] + max_range)
            ylim = (ylim[0], ylim[0] + max_range)

        print("plot video")
        FFMpegWriter = animation.writers["ffmpeg"]
        metadata = dict(title="Movie Test",
                        artist="Matplotlib",
                        comment="Movie support!")
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
        ax = fig.add_subplot(111)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("z [m]", fontsize=16)
        ax.set_ylabel("x [m]", fontsize=16)
        rod_lines_2d = ax.plot(positions_over_time[0][2],
                               positions_over_time[0][0])[0]

        # 初始化一个变量来保存当前的圆形对象
        object_plots = [None for _ in range(len(self.objects))]

        with writer.saving(fig, video_name, dpi=150):
            for time in tqdm(range(1, positions_over_time.shape[0])):
                # 更新杆的位置
                rod_lines_2d.set_xdata(positions_over_time[time][2])
                rod_lines_2d.set_ydata(positions_over_time[time][0])

                # 移除旧的圆形（如果存在）
                for idx, obj in enumerate(self.objects):
                    if object_plots[idx] is not None:
                        object_plots[idx].remove()

                    if isinstance(obj, ea.Sphere):
                        # 添加新的圆形
                        center_x = object_positions[idx][time][2]
                        center_y = object_positions[idx][time][0]
                        radius = obj.radius[0]
                        color = 'red' if target_last and idx == len(
                            self.objects) - 1 else 'lightblue'
                        object_plots[idx] = plt.Circle((center_x, center_y),
                                                       radius,
                                                       edgecolor='b',
                                                       facecolor=color)
                        ax.add_patch(object_plots[idx])

                # 捕捉当前帧
                writer.grab_frame()

        # 关闭图形
        plt.close(plt.gcf())

    def visualize_3d(self, video_name, fps, xlim=None, ylim=None, zlim=None):
        rod_objects_3d_visualize(
            np.array(self.rod_callback["position"]),
            [np.array(params["position"]) for params in self.object_callbacks],
            self.objects,
            video_name,
            fps,
            1,
            xlim,
            ylim,
            zlim,
        )

    def visualize_3d_povray(
        self,
        video_name,
        output_images_dir,
        fps,
        width=1920,
        height=1080,
    ):
        xz_positions = [
            np.array(callback["position"])[:, [0, 2], 0]
            for callback in self.object_callbacks
        ]
        xz_positions = np.concatenate(xz_positions, axis=0)
        x_min, z_min = np.min(xz_positions, axis=0)
        x_max, z_max = np.max(xz_positions, axis=0)
        x_mid, z_mid = (x_min + x_max) / 2, (z_min + z_max) / 2

        renderer = POVRayRenderer(
            output_filename=video_name,
            output_images_dir=output_images_dir,
            fps=fps,
            width=width,
            height=height,
            top_camera_position=[x_mid, 20, z_mid],
            top_camera_look_at=[x_mid, 0, z_mid],
        )

        frames = len(self.rod_callback['time'])
        for i in tqdm(range(frames), disable=False, desc="Rendering .povray"):
            renderer.reset_stage(
                top_camera_position=[x_mid, 10, z_mid],
                top_camera_look_at=[x_mid, 0, z_mid],
            )
            for object_ in self.objects:
                id_ = self.object2id[object_]
                object_callback = self.object_callbacks[id_]
                if isinstance(object_, ea.Sphere):
                    renderer.add_stage_object(
                        object_type='sphere',
                        name=f'sphere{id_}',
                        position=np.squeeze(object_callback['position'][i]),
                        radius=np.squeeze(object_callback['radius'][i]),
                    )
                elif isinstance(object_, MeshSurface):
                    renderer.add_stage_object(
                        object_type='mesh',
                        name=f'mesh{id_}',
                        mesh_name='cube_mesh',
                        position=np.squeeze(object_callback['position'][i]),
                        scale=0.5,  # TODO
                        matrix=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                    )
            # start = time.time()
            renderer.render_single_step(
                data={
                    "rod_position": self.rod_callback["position"][i],
                    "rod_radius": self.rod_callback["radius"][i],
                }
                # save_img=True,
            )
            # end = time.time()
            # print("Render time per render step: ", end - start)
        renderer.process_povray(multi_processing=True)
        renderer.create_video()


class RodObjectsEnvironment(SimulateMixin, RodObjectsMixin, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def setup(self):
        pass

    def add_objects(self, object_configs: list):
        for object_config in object_configs:
            if isinstance(object_config, SphereArguments):
                self.add_sphere(
                    center=object_config.center,
                    radius=object_config.radius,
                    density=object_config.density,
                )
            elif isinstance(object_config, MeshSurfaceArguments):
                self.add_mesh_surface(
                    mesh_path=object_config.mesh_path,
                    center=object_config.center,
                    scale=object_config.scale,
                    rotate=object_config.rotate,
                )
            else:
                raise ValueError(
                    f"Unsupported object type: {type(object_config)}")
