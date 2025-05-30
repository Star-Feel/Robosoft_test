__all__ = [
    "SimulateMixin",
    "RodMixin",
    "RigidMixin",
    "ObjectsMixin",
    "FetchableRodObjectsEnvironment",
    "RodSphereEnvironment",
    "RodControlMixin",
    "SimulatedEnvironment",
]

import os
import pickle
from abc import ABC, abstractmethod
from typing import Optional

import elastica as ea
import gym
import matplotlib.pyplot as plt
import numpy as np
from elastica import (
    BaseSystemCollection,
    CallBacks,
    Connections,
    Constraints,
    Contact,
    Damping,
    Forcing,
    PositionVerlet,
)
from elastica.timestepper import extend_stepper_interface
from gym import spaces
from matplotlib import animation
from tqdm import tqdm

from ..components import MeshSurface, RigidBodyCallBack, RodCallBack
from ..components.callback import MeshSurfaceCallBack
from ..utils import compute_rotation_matrix
from ..visualize.pov2blend import BlenderRenderer
from ..visualize.renderer import POVRayRenderer
from ..visualize.visualizer import rod_objects_3d_visualize


class BaseSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, Damping, Contact,
    CallBacks
):
    """Base simulator class combining Elastica functionalities."""
    pass


class SimulatedEnvironment(ABC):

    def __init__(
        self,
        *args,
        final_time: float,
        time_step: int,
        update_interval: int = 1,
        rendering_fps: int = 60,
        **kwargs
    ):

        self.simulator = BaseSimulator()

        self.final_time = final_time
        self.time_step = time_step
        self.update_interval = update_interval
        self.rendering_fps = rendering_fps
        self.stateful_stepper = PositionVerlet()
        self.total_steps = int(final_time / time_step)
        self.time_step = np.float64(float(final_time) / self.total_steps)
        self.time_tracker = np.float64(0.0)

        super().__init__(*args, **kwargs)

    @abstractmethod
    def setup(self):
        pass

    def reset_simulator(self):
        self.simulator = BaseSimulator()
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
            self.stateful_stepper, self.simulator
        )


class SimulateMixin(SimulatedEnvironment):
    """
    Deprecated: Use SimulatedEnvironment instead.
    """

    def __init__(
        self,
        *args,
        final_time: float,
        time_step: int,
        update_interval: int = 1,
        rendering_fps: int = 60,
        **kwargs
    ):

        self.final_time = final_time
        self.time_step = time_step
        self.update_interval = update_interval
        self.rendering_fps = rendering_fps
        self.stateful_stepper = PositionVerlet()
        self.total_steps = int(final_time / time_step)
        self.time_step = np.float64(float(final_time) / self.total_steps)
        self.time_tracker = np.float64(0.0)

        super().__init__(*args, **kwargs)

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
            self.stateful_stepper, self.simulator
        )


class RodMixin(SimulatedEnvironment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rod_callback = ea.defaultdict(list)

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

    def _add_data_collection_callbacks(self, step_skip: int):
        self.simulator.collect_diagnostics(self.shearable_rod).using(
            RodCallBack,
            step_skip=step_skip,
            callback_params=self.rod_callback
        )


class RigidMixin(SimulatedEnvironment):

    def add_sphere(
        self,
        center: np.ndarray,
        radius: float,
        density: float,
        theta: np.ndarray,
    ) -> ea.Sphere:
        """
        Add a sphere to the environment.

        Args:
            center (np.ndarray): Center of the sphere.
            radius (float): Radius of the sphere.
        """
        sphere = ea.Sphere(center, radius, density=density)
        rotate_matrix = compute_rotation_matrix(theta)
        sphere.director_collection[..., 0] = rotate_matrix
        self.simulator.append(sphere)
        return sphere

    def add_mesh_surface(
        self,
        mesh_path: str,
        center: np.ndarray = np.array([0., 0., 0.]),
        scale: np.ndarray = np.array([1., 1., 1.]),
        rotate: np.ndarray = np.array([0., 0., 0.]),
    ) -> MeshSurface:
        mesh = MeshSurface(mesh_path)
        mesh.scale(scale)
        mesh.rotate(np.array([1, 0, 0]), rotate[0])
        mesh.rotate(np.array([0, 1, 0]), rotate[1])
        mesh.rotate(np.array([0, 0, 1]), rotate[2])
        mesh.translate(center)
        self.simulator.append(mesh)
        return mesh


class ObjectsMixin(RigidMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objects = []
        self.object2id = {}
        self.object_callbacks = []

    def add_sphere(
        self,
        center: np.ndarray,
        radius: float,
        density: float,
        theta: np.ndarray,
    ) -> ea.Sphere:
        """
        Add a sphere to the environment.

        Args:
            center (np.ndarray): Center of the sphere.
            radius (float): Radius of the sphere.
        """
        sphere = super().add_sphere(center, radius, density, theta)
        self.objects.append(sphere)
        self.object2id[sphere] = len(self.objects) - 1
        self.object_callbacks.append(ea.defaultdict(list))
        return sphere

    def add_mesh_surface(
        self,
        mesh_path: str,
        center: np.ndarray = np.array([0., 0., 0.]),
        scale: np.ndarray = np.array([1., 1., 1.]),
        rotate: np.ndarray = np.array([0., 0., 0.]),
    ) -> MeshSurface:
        mesh = super().add_mesh_surface(mesh_path, center, scale, rotate)
        self.objects.append(mesh)
        self.object2id[mesh] = len(self.objects) - 1
        self.object_callbacks.append(ea.defaultdict(list))
        return mesh

    def add_cylinder(self):
        pass

    def _add_data_collection_callbacks(self, step_skip: int):
        for object_ in self.objects:
            if isinstance(object_, ea.RigidBodyBase):
                self.simulator.collect_diagnostics(object_).using(
                    RigidBodyCallBack,
                    step_skip=step_skip,
                    callback_params=self.object_callbacks[
                        self.object2id[object_]]
                )
            elif isinstance(object_, MeshSurface):
                self.simulator.collect_diagnostics(object_).using(
                    MeshSurfaceCallBack,
                    step_skip=step_skip,
                    callback_params=self.object_callbacks[
                        self.object2id[object_]]
                )


class FetchableRodObjectsEnvironment(
    RodMixin, ObjectsMixin, SimulatedEnvironment
):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_flags = []
        self.attach_flags = []

    def add_sphere(
        self,
        center: np.ndarray,
        radius: float,
        density: float,
        theta: np.ndarray,
    ) -> ea.Sphere:
        """
        Add a sphere to the environment.

        Args:
            center (np.ndarray): Center of the sphere.
            radius (float): Radius of the sphere.
        """
        if theta is None:
            theta_x = 0
            theta_y = np.random.uniform(np.pi / 6, np.pi / 3)
            theta_z = 0
            theta = np.array([theta_x, theta_y, theta_z])

        sphere = super().add_sphere(center, radius, density, theta)

        self.action_flags.append(False)
        self.attach_flags.append(False)

        return sphere

    def add_cylinder(self):
        pass

    def add_mesh_surface(
        self,
        mesh_path: str,
        center: np.ndarray = np.array([0., 0., 0.]),
        scale: np.ndarray = np.array([1., 1., 1.]),
        rotate: np.ndarray = np.array([0., 0., 0.]),
    ) -> MeshSurface:
        mesh = super().add_mesh_surface(mesh_path, center, scale, rotate)

        self.action_flags.append(False)
        self.attach_flags.append(False)

        return mesh

    def _add_data_collection_callbacks(self, step_skip: int):
        RodMixin._add_data_collection_callbacks(self, step_skip)
        ObjectsMixin._add_data_collection_callbacks(self, step_skip)

    def export_callbacks(self, filename):
        """
        Export the collected callback data to a file.

        Args:
            filename (str): The name of the file to save the callback data.
        """

        callback_data = {
            "deciption": "rod is the softrobot, whose callback is a dict of positions,"
            "velocities... \nobjects is all spheres, whose callback is a list"
            "of each object's callback",
            "rod_callback": self.rod_callback,
            "object_callbacks": self.object_callbacks,
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
        metadata = dict(
            title="Movie Test", artist="Matplotlib", comment="Movie support!"
        )
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
        ax = fig.add_subplot(111)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("z [m]", fontsize=16)
        ax.set_ylabel("x [m]", fontsize=16)
        rod_lines_2d = ax.plot(
            positions_over_time[0][2], positions_over_time[0][0]
        )[0]

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
                            self.objects
                        ) - 1 else 'lightblue'
                        object_plots[idx] = plt.Circle((center_x, center_y),
                                                       radius,
                                                       edgecolor='b',
                                                       facecolor=color)
                        ax.add_patch(object_plots[idx])
                    elif isinstance(obj, MeshSurface):

                        # 添加新的圆点
                        center_x = object_positions[idx][time][2]
                        center_y = object_positions[idx][time][0]
                        color = 'red' if target_last and idx == len(
                            self.objects
                        ) - 1 else 'lightblue'
                        object_plots[idx] = ax.plot(
                            center_x, center_y, 'o', color=color
                        )[0]

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

    def visualize_3d_blender(
        self,
        video_name,
        output_images_dir,
        fps,
        width=960,
        height=540,
        target_id=0,
    ):
        top_view_dir = os.path.join(output_images_dir, "top")
        blender_renderer = BlenderRenderer(top_view_dir)

        renderer = POVRayRenderer(
            output_filename=video_name,
            output_images_dir=output_images_dir,
            fps=fps,
            width=width,
            height=height,
        )

        # refine camera setting
        # x_max = max(list(point.center[0] for point in self.object_configs))
        # y_max = max(list(point.center[1] for point in self.object_configs))
        # z_max = max(list(point.center[2] for point in self.object_configs))

        # x_min = min(list(point.center[0] for point in self.object_configs))
        # y_min = min(list(point.center[1] for point in self.object_configs))
        # z_min = min(list(point.center[2] for point in self.object_configs))

        # x_avg = (x_max + x_min) / 2
        # y_avg = (y_max + y_min) / 2
        # z_avg = (z_max + z_min) / 2

        frames = len(self.rod_callback['time'])
        for i in tqdm(range(frames), disable=False, desc="Rendering .povray"):
            renderer.reset_stage(
                top_camera_position=[2, 7, 1], top_camera_look_at=[0, 0, 1]
            )
            for object_ in self.objects:
                id_ = self.object2id[object_]
                object_callback = self.object_callbacks[id_]
                object_name = "target_object" if id_ == target_id else "obstacle_object"
                if isinstance(object_, ea.Sphere):
                    renderer.add_stage_object(
                        object_type='sphere',
                        name=f'sphere{id_}',
                        shape=str(self.object_configs[id_].shape),
                        object_name=object_name,
                        position=np.squeeze(object_callback['position'][i]),
                        radius=np.squeeze(object_callback['radius'][i]),
                    )
                elif isinstance(object_, MeshSurface):
                    scale = np.linalg.norm(object_.mesh_scale)
                    renderer.add_stage_object(
                        object_type='mesh',
                        name=f'mesh{id_}',
                        shape=str(self.object_configs[id_].shape),
                        object_name=object_name,
                        position=np.squeeze(object_callback['position'][i]),
                        scale=scale,
                        matrix=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                    )
            renderer.render_single_step(
                data={
                    "rod_position": self.rod_callback["position"][i],
                    "rod_radius": self.rod_callback["radius"][i],
                },
                save_script_file=True,
                save_img=False,
            )

        blender_renderer.batch_rendering(top_view_dir, top_view_dir)
        renderer.create_video(only_top=True)

    def single_step_3d_blend(
        self,
        output_images_dir,
        fps,
        width=960,
        height=540,
        current_step=0,
        interval=0,
        target_id=0,
        save_img: bool = False,
    ):
        if current_step % interval == 0:
            top_view_dir = os.path.join(output_images_dir, "top")
            blender_renderer = BlenderRenderer(top_view_dir)

            renderer = POVRayRenderer(
                output_images_dir=output_images_dir,
                fps=fps,
                width=width,
                height=height,
            )

            renderer.reset_stage(
                top_camera_position=[2, 7, 1], top_camera_look_at=[0, 0, 1]
            )
            for object_ in self.objects:
                id_ = self.object2id[object_]
                object_callback = self.object_callbacks[id_]
                if isinstance(object_, ea.Sphere):
                    renderer.add_stage_object(
                        object_type='sphere',
                        name=f'sphere{id_}',
                        shape=str(self.object_configs[id_].shape),
                        position=np.squeeze(object_callback['position'][0]),
                        radius=np.squeeze(object_callback['radius'][0]),
                    )
                elif isinstance(object_, MeshSurface):
                    scale = np.linalg.norm(object_.mesh_scale)
                    renderer.add_stage_object(
                        object_type='mesh',
                        name=f'mesh{id_}',
                        shape=str(self.object_configs[id_].shape),
                        mesh_name='cube_mesh',
                        position=np.squeeze(object_callback['position'][0]),
                        scale=scale,  # TODO
                        matrix=[1, 0, 0, 0, 1, 0, 0, 0, 1],
                    )

            pov_scripts = renderer.render_single_step(
                data={
                    "rod_position": self.shearable_rod.position_collection,
                    "rod_radius": self.shearable_rod.radius,
                },
                save_img=False,
            )

            rendered_image = blender_renderer.single_step_rendering(
                current_step,
                pov_scripts["top"],
                top_view_dir,
                save_img,
            )


class RodSphereEnvironment(RodMixin, RigidMixin, SimulatedEnvironment):

    shearable_rod: ea.CosseratRod
    sphere: ea.Sphere

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rod_callback = ea.defaultdict(list)
        self.sphere_callback = ea.defaultdict(list)

    def _add_data_collection_callbacks(self, step_skip: int):
        RodMixin._add_data_collection_callbacks(self, step_skip)
        self.simulator.collect_diagnostics(self.sphere).using(
            RigidBodyCallBack,
            step_skip=step_skip,
            callback_params=self.sphere_callback
        )

    def visualize_3d(self, video_name, fps, xlim=None, ylim=None, zlim=None):
        rod_objects_3d_visualize(
            np.array(self.rod_callback["position"]),
            [
                np.array(params["position"])
                for params in [self.sphere_callback]
            ],
            [self.sphere],
            video_name,
            fps,
            1,
            xlim,
            ylim,
            zlim,
        )


class RodControlMixin(SimulatedEnvironment, gym.Env):

    def __init__(
        self,
        *args,
        number_of_control_points: int,
        n_elem: int = 40,
        obs_state_points: int = 10,
        trainable: bool = False,
        boundary: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_elem = n_elem
        self.number_of_control_points = number_of_control_points
        self.obs_state_points = obs_state_points
        self.trainable = trainable

        # normal, binormal and/or tangent direction activation (3D)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3 * self.number_of_control_points, ),
            dtype=np.float64,
        )
        self.action = np.zeros(3 * self.number_of_control_points)

        # num_points = int(n_elem / self.obs_state_points)
        # num_rod_state = len(np.ones(n_elem + 1)[0::num_points])

        if self.trainable:
            self.total_learning_steps = int(
                self.total_steps / self.update_interval
            )
            self.boundary = boundary
            self.trainable_init()

    def trainable_init(self):
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3 * self.number_of_control_points, ),
            dtype=np.float64,
        )
        self.action = np.zeros(3 * self.number_of_control_points)

        num_points = int(self.n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(self.n_elem + 1)[0::num_points])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_rod_state * 3 + 8 + 11, ),
            dtype=np.float64,
        )

    def reset(self):
        self.simulator = BaseSimulator()
        self.time_tracker = np.float64(0.0)
        state = self.setup()
        return state

    def get_state(self):
        pass

    def render(self):
        pass

    def step(self):
        pass

    def sampleAction(self):

        random_action = (
            np.random.rand(1 * self.number_of_control_points) - 0.5
        ) * 2
        return random_action

    def seed(self, seed):
        np.random.seed(seed)
