__all__ = [
    "NavigationSnakeActionEnvironment",
    "NavigationSnakeTorqueEnvironment",
    "NavigationSnakeArguments",
    "NavigationSnakeTorqueEnvironmentForGymTrain",
]
from dataclasses import dataclass
from typing import Optional

import elastica as ea
import numpy as np
from elastica import RigidBodyBase
from elastica._calculus import _isnan_check
import gym

from ..arguments import (
    RodArguments,
    SimulatorArguments,
    SphereArguments,
    SuperArguments,
)
from ..components import (
    ChangeableMuscleTorques,
    RigidBodyAnalyticalLinearDamper,
    RodMeshSurfaceContactWithGridMethod,
)
from ..components.contact import JoinableRodSphereContact, surface_grid_xyz
from ..components.surface.mesh_surface import MeshSurface
from .base_envs import BaseSimulator, FetchableRodObjectsEnvironment, SimulatedEnvironment


@dataclass
class NavigationSnakeArguments(SuperArguments):

    rod: RodArguments
    objects: list[SphereArguments]
    simulator: SimulatorArguments


class NavigationSnakeActionEnvironment(FetchableRodObjectsEnvironment):

    def __init__(self, configs: NavigationSnakeArguments):
        self.rod_config = configs.rod
        self.sim_config = configs.simulator
        self.object_configs = configs.objects if configs.objects else []

        super().__init__(
            final_time=self.sim_config.final_time,
            time_step=self.sim_config.time_step,
            update_interval=self.sim_config.update_interval,
            rendering_fps=self.sim_config.rendering_fps,
        )

        self.turn = [ChangeableMuscleTorques.DIRECT]
        self.torque_callback = []

    def setup(self, callback_step_skip: int = -1):
        shear_modulus = self.rod_config.youngs_modulus / (
            self.rod_config.poisson_ratio + 1.0
        )

        self.add_shearable_rod(
            n_elem=self.rod_config.n_elem,
            start=self.rod_config.start,
            direction=self.rod_config.direction,
            normal=self.rod_config.normal,
            base_length=self.rod_config.base_length,
            base_radius=self.rod_config.base_radius,
            density=self.rod_config.density,
            youngs_modulus=self.rod_config.youngs_modulus,
            shear_modulus=shear_modulus,
        )
        b_coeff = np.array([17.4, 48.5, 5.4, 14.7, 0.97])
        gravitational_acc = -9.80665
        period = 0.5

        for object_config in self.object_configs:
            if isinstance(object_config, SphereArguments):
                self.add_sphere(
                    center=object_config.center,
                    radius=object_config.radius,
                    density=object_config.density,
                )

        # Add gravitational forces
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ea.GravityForces,
            acc_gravity=np.array([0.0, gravitational_acc, 0.0]),
        )

        wave_length = b_coeff[-1]
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ChangeableMuscleTorques,
            turn=self.turn,
            callbacks=self.torque_callback,
            base_length=self.rod_config.base_length,
            b_coeff=b_coeff[:-1],
            period=period,
            wave_number=2.0 * np.pi / (wave_length),
            phase_shift=0.0,
            direction=self.rod_config.normal,
            rest_lengths=self.shearable_rod.rest_lengths,
            ramp_up_time=period,
            with_spline=True,
        )

        # Add friction forces
        origin_plane = np.array([0.0, -self.rod_config.base_radius, 0.0])
        normal_plane = self.rod_config.normal
        slip_velocity_tol = 1e-8
        froude = 0.1
        mu = self.rod_config.base_length / (
            period * period * np.abs(gravitational_acc) * froude
        )
        kinetic_mu_array = np.array([mu, 8 * mu, 20.0 * mu]
                                    )  # [forward, backward, sideways]
        static_mu_array = 2 * kinetic_mu_array
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ea.AnisotropicFrictionalPlane,
            k=1,
            nu=1e-6,
            plane_origin=origin_plane,
            plane_normal=normal_plane,
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )

        # Add damping
        damping_constant = 4.0
        self.simulator.dampen(self.shearable_rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )

        callback_step_skip = int(
            1.0 / (self.sim_config.rendering_fps * self.time_step)
        ) if callback_step_skip < 0 else callback_step_skip
        # Add callbacks for data collection if enabled
        self._add_data_collection_callbacks(callback_step_skip)

        self._finalize()

        return self

    def step(self, num_steps: int = 1):
        for _ in range(num_steps):
            self._do_step()

            if _isnan_check(self.shearable_rod.position_collection):
                print("NaN detected, simulation is unstable")
                return {"status": "error", "message": "NaN values detected"}

        return {
            "time": self.time_tracker,
            "rod_tip_position": self.shearable_rod.position_collection[
                ..., -1].copy(),
            "status": "ok"
        }

    def set_target(self, target: list[float]):
        self.target = np.array(target)

    def reach(self, eps=0.01):
        rod_tip_position = self.shearable_rod.position_collection[...,
                                                                  -1].copy()
        distance = np.linalg.norm(rod_tip_position - self.target)
        return distance < eps


class NavigationSnakeTorqueEnvironment(FetchableRodObjectsEnvironment):

    def __init__(self, configs: NavigationSnakeArguments):
        self.rod_config = configs.rod
        self.sim_config = configs.simulator
        self.object_configs = configs.objects if configs.objects else []

        super().__init__(
            final_time=self.sim_config.final_time,
            time_step=self.sim_config.time_step,
            update_interval=self.sim_config.update_interval,
            rendering_fps=self.sim_config.rendering_fps,
        )

        self.torque = np.zeros((3, self.rod_config.n_elem))

    def add_rod_objects_contact(self):
        for obj in self.objects:
            collision = not (self.object2id[obj] == self.target_id)
            if isinstance(obj, ea.Sphere):
                self.simulator.detect_contact_between(
                    self.shearable_rod, obj
                ).using(
                    JoinableRodSphereContact,
                    k=50,
                    nu=10,
                    velocity_damping_coefficient=1e3,
                    friction_coefficient=10,
                    index=np.array(range(self.rod_config.n_elem + 1)),
                    action_flags=self.action_flags,
                    attach_flags=self.attach_flags,
                    flag_id=self.object2id[obj],
                    collision=collision
                )
            elif isinstance(obj, MeshSurface):
                grid_size = np.min(obj.mesh_scale) / 10
                # faces: (dim, n_faces, n_points)
                faces_grid = surface_grid_xyz(obj.faces, grid_size)
                faces_grid["model_path"] = obj.model_path
                faces_grid["grid_size"] = grid_size
                faces_grid["surface_reorient"] = obj.mesh_orientation
                k = 1e3  # 接触刚度系数
                nu = 100  # 阻尼系数
                surface_tol = 1e-6
                self.simulator.detect_contact_between(
                    self.shearable_rod, obj
                ).using(
                    RodMeshSurfaceContactWithGridMethod,
                    k=k,
                    nu=nu,
                    faces_grid=faces_grid,
                    grid_size=grid_size,
                    surface_tol=surface_tol
                )

    def add_dampen_to_objects(self, dampen_constant: float = 1):
        for object_ in self.objects:
            if isinstance(object_, RigidBodyBase):
                self.simulator.dampen(object_).using(
                    RigidBodyAnalyticalLinearDamper,
                    damping_constant=dampen_constant,
                    time_step=self.time_step,
                )

    def setup(self, target_id: int = -1, callback_step_skip: int = -1):
        self.target_id = target_id
        # if self.target_id >= 0:
        #     self.set_target(self.target_id)

        shear_modulus = self.rod_config.youngs_modulus / (
            self.rod_config.poisson_ratio + 1.0
        )

        self.add_shearable_rod(
            n_elem=self.rod_config.n_elem,
            start=self.rod_config.start,
            direction=self.rod_config.direction,
            normal=self.rod_config.normal,
            base_length=self.rod_config.base_length,
            base_radius=self.rod_config.base_radius,
            density=self.rod_config.density,
            youngs_modulus=self.rod_config.youngs_modulus,
            shear_modulus=shear_modulus,
        )
        b_coeff = np.array([17.4, 48.5, 5.4, 14.7, 0.97])
        gravitational_acc = -9.80665
        period = 0.5

        self.add_objects(self.object_configs)

        self.add_rod_objects_contact()
        self.add_dampen_to_objects(1e8)

        # Add gravitational forces
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ea.GravityForces,
            acc_gravity=np.array([0.0, gravitational_acc, 0.0]),
        )

        wave_length = b_coeff[-1]
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ChangeableMuscleTorques,
            torque=self.torque,
            base_length=self.rod_config.base_length,
            b_coeff=b_coeff[:-1],
            period=period,
            wave_number=2.0 * np.pi / (wave_length),
            phase_shift=0.0,
            direction=self.rod_config.normal,
            rest_lengths=self.shearable_rod.rest_lengths,
            ramp_up_time=period,
            with_spline=True,
        )

        # Add friction forces
        origin_plane = np.array([0.0, -self.rod_config.base_radius, 0.0])
        normal_plane = self.rod_config.normal
        slip_velocity_tol = 1e-8
        froude = 0.1
        mu = self.rod_config.base_length / (
            period * period * np.abs(gravitational_acc) * froude
        )
        kinetic_mu_array = np.array([mu, 8 * mu, 20.0 * mu]
                                    )  # [forward, backward, sideways]
        static_mu_array = 2 * kinetic_mu_array
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ea.AnisotropicFrictionalPlane,
            k=1,
            nu=1e-6,
            plane_origin=origin_plane,
            plane_normal=normal_plane,
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )

        # Add damping
        damping_constant = 4.0
        self.simulator.dampen(self.shearable_rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )

        callback_step_skip = int(
            1.0 / (self.sim_config.rendering_fps * self.time_step)
        ) if callback_step_skip < 0 else callback_step_skip
        # Add callbacks for data collection if enabled
        self._add_data_collection_callbacks(callback_step_skip)

        self._finalize()

        return self

    def step(self, torque: np.ndarray, num_steps: int = 1):

        self.torque[:] = torque[:]

        for _ in range(num_steps):
            self._do_step()

            if _isnan_check(self.shearable_rod.position_collection):
                print("NaN detected, simulation is unstable")
                return {"status": "error", "message": "NaN values detected"}

        return {
            "time": self.time_tracker,
            "rod_tip_position": self.shearable_rod.position_collection[
                ..., -1].copy(),
            "status": "ok"
        }

    def set_target(self, target_id: int):
        self.target = self.object_configs[target_id].center
        self.eps = self.object_configs[target_id].radius

    def set_target_id(self, target_id: int):
        self.target_id = target_id

    def reach(self, eps=None):
        eps = eps if eps is not None else self.eps
        rod_tip_position = self.shearable_rod.position_collection[...,
                                                                  -1].copy()
        distance = np.linalg.norm(
            rod_tip_position[[0, 2]] - self.target[[0, 2]]
        )
        return distance < eps


class RodControlMixin(SimulatedEnvironment, gym.Env):

    def __init__(
        self,
        *args,
        trainable: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trainable = trainable

        if self.trainable:
            self.trainable_init()

    def trainable_init(self):
        pass

    def reset(self):
        self.simulator = BaseSimulator()
        self.time_tracker = np.float64(0.0)
        state = self.setup()
        return state

    def get_state(self):
        pass

    def render(self):
        pass

    def step(self, action: np.ndarray):
        pass

    def sampleAction(self):

        random_action = (
            np.random.rand(1 * self.number_of_control_points) - 0.5
        ) * 2
        return random_action

    def seed(self, seed):
        np.random.seed(seed)


class NavigationSnakeTorqueEnvironmentForGymTrain(
    NavigationSnakeTorqueEnvironment,
    RodControlMixin,
):

    def __init__(self, configs: NavigationSnakeArguments):
        super().__init__(configs)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3, self.rod_config.n_elem),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8,
        )

    def setup(self):
        self.objects = []
        self.object2id = {}
        self.object_callbacks = []
        self.action_flags = []
        self.attach_flags = []
        super().setup()
        state = self.get_state()
        return state

    def step(self, action: np.ndarray):
        super().step(action, 1)
        state = self.get_state()
        reward = np.random.randint(0, 10)
        if reward > 5:
            done = True
        else:
            done = False

        return state, reward, done, {"ctime": self.time_tracker}

    def get_state(self):
        # 渲染出图像
        state = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        return state
