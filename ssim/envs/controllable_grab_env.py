__all__ = [
    "ControllableGrabArguments",
    "ControllableGrabEnvironment",
]

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import elastica as ea
import numpy as np
from elastica import OneEndFixedRod, RigidBodyBase

from ..arguments import (
    MeshSurfaceArguments,
    RodArguments,
    RodControllerArgumets,
    SimulatorArguments,
    SphereArguments,
    SuperArguments,
)
from ..components import (
    JoinableRodSphereContact,
    MuscleTorquesWithVaryingBetaSplines,
    RigidBodyAnalyticalLinearDamper,
    RodMeshSurfaceContactWithGridMethod,
)
from ..components.contact import surface_grid_xyz
from ..components.surface.mesh_surface import MeshSurface
from ..utils import (
    compute_quaternion_from_matrix,
    compute_rotation_matrix,
    isnan_check,
)
from .base_envs import (
    FetchableRodObjectsEnvironment,
    RodControlMixin,
    SimulatedEnvironment,
)


@dataclass
class ControllableGrabArguments(SuperArguments):

    rod: RodArguments
    objects: Sequence[SphereArguments | MeshSurfaceArguments] = ()
    simulator: SimulatorArguments = None
    controller: RodControllerArgumets = None


class ControllableGrabEnvironment(
    FetchableRodObjectsEnvironment, RodControlMixin, SimulatedEnvironment
):

    def __init__(self, configs: ControllableGrabArguments):

        self.rod_config = configs.rod
        self.object_configs = configs.objects
        self.sim_config = configs.simulator
        self.control_config = configs.controller

        super().__init__(
            final_time=self.sim_config.final_time,
            time_step=self.sim_config.time_step,
            update_interval=self.sim_config.update_interval,
            rendering_fps=self.sim_config.rendering_fps,
            number_of_control_points=self.control_config.
            number_of_control_points,
            n_elem=self.rod_config.n_elem,
            obs_state_points=self.control_config.obs_state_points,
            trainable=self.control_config.trainable,
            boundary=self.control_config.boundary,
        )

        self.alpha = 75
        self.beta = 75
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        self.spline_points_func_array_normal_dir = []
        # self.torque_normal_callback = []

        self.torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)
        self.spline_points_func_array_binormal_dir = []
        # self.torque_binormal_callback = []

        self.torque_profile_list_for_muscle_in_twist_dir = defaultdict(list)
        self.spline_points_func_array_twist_dir = []
        # self.torque_twist_callback = []

        self.post = []

        self.target_point = None
        self.target_angle = None

        self.current_step = 0

    def setup(self):

        shear_modulus = self.rod_config.youngs_modulus / (
            self.rod_config.poisson_ratio + 1.0
        )

        radius_along_rod = np.linspace(
            self.rod_config.base_radius, self.rod_config.radius_tip,
            self.rod_config.n_elem
        )
        self.add_shearable_rod(
            n_elem=self.rod_config.n_elem,
            start=self.rod_config.start,
            direction=self.rod_config.direction,
            normal=self.rod_config.normal,
            base_length=self.rod_config.base_length,
            base_radius=radius_along_rod,
            density=self.rod_config.density,
            youngs_modulus=self.rod_config.youngs_modulus,
            shear_modulus=shear_modulus,
        )

        for object_config in self.object_configs:
            if isinstance(object_config, SphereArguments):
                self.add_sphere(
                    center=object_config.center,
                    radius=object_config.radius,
                    density=object_config.density,
                    theta=None,
                )
            elif isinstance(object_config, MeshSurfaceArguments):
                self.add_mesh_surface(
                    object_config.mesh_path, object_config.center,
                    object_config.scale, object_config.rotate
                )

        # fix one end of the rod
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod,
            constrained_position_idx=(0, ),
            constrained_director_idx=(0, )
        )

        callback_step_skip = int(
            1.0 / (self.sim_config.rendering_fps * self.time_step)
        )
        # Add muscle torques acting on the arm for actuation
        # MuscleTorquesWithVaryingBetaSplines uses the control points selected
        # by RL to generate torques along the arm.
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.rod_config.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_normal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("normal"),
            step_skip=callback_step_skip,
            max_rate_of_change_of_activation=np.inf,
            torque_profile_recorder=self.
            torque_profile_list_for_muscle_in_normal_dir,
            # callbacks=self.torque_normal_callback,
        )

        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.rod_config.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_binormal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("binormal"),
            step_skip=callback_step_skip,
            max_rate_of_change_of_activation=np.inf,
            torque_profile_recorder=self.
            torque_profile_list_for_muscle_in_binormal_dir,
            # callbacks=self.torque_binormal_callback,
        )

        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.rod_config.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_twist_dir,
            muscle_torque_scale=self.beta,
            direction=str("tangent"),
            step_skip=callback_step_skip,
            max_rate_of_change_of_activation=np.inf,
            torque_profile_recorder=self.
            torque_profile_list_for_muscle_in_twist_dir,
            # callbacks=self.torque_twist_callback,
        )

        # damping of rod and objects
        damping_constant = 5
        self.simulator.dampen(self.shearable_rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )

        for obj in self.objects:
            if isinstance(obj, ea.Sphere):
                self.simulator.detect_contact_between(
                    self.shearable_rod, obj
                ).using(
                    JoinableRodSphereContact,
                    k=10,
                    nu=30,
                    velocity_damping_coefficient=1e3,
                    friction_coefficient=10,
                    action_flags=self.action_flags,
                    attach_flags=self.attach_flags,
                    flag_id=self.object2id[obj],
                    collision=False,
                    eps=1
                )
            elif isinstance(obj, MeshSurface):
                grid_size = np.min(obj.mesh_scale) / 10
                # faces: (dim, n_faces, n_points)
                faces_grid = surface_grid_xyz(obj.faces, grid_size)
                faces_grid["model_path"] = obj.model_path
                faces_grid["grid_size"] = grid_size
                faces_grid["surface_reorient"] = obj.mesh_orientation
                self.simulator.detect_contact_between(
                    self.shearable_rod, obj).using(
                        RodMeshSurfaceContactWithGridMethod,
                        k=1e4,  # 接触刚度系数
                        nu=30,  # 阻尼系数
                        faces_grid=faces_grid,
                    grid_size=grid_size,
                        surface_tol=1e-2,
                    )

        dampen = 10000
        for _, object_ in enumerate(self.objects):
            if isinstance(object_, RigidBodyBase):
                self.simulator.dampen(object_).using(
                    RigidBodyAnalyticalLinearDamper,
                    damping_constant=dampen,
                    time_step=self.time_step,
                )

        # Add callbacks for data collection if enabled
        self._add_data_collection_callbacks(callback_step_skip)

        self._finalize()

        # state = self.get_state()

    def set_target(self, target_point, target_angle):
        """
        Set target point and angle for the rod.

        Parameters
        ----------
        target_point : numpy.ndarray
            Target point in 3D space.
        target_angle : numpy.ndarray
            Target angle in 3D space.
        """

        self.target_point = target_point
        self.target_angle = target_angle

        return self.get_state()

    def get_rod_state(self):

        position = self.shearable_rod.position_collection
        velocity = self.shearable_rod.velocity_collection

        return copy.deepcopy(position), copy.deepcopy(velocity)

    def get_state(self):
        """
        Returns current state of the system to the controller.

        Returns
        -------
        numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        """

        rod_state = self.shearable_rod.position_collection
        r_s_a = rod_state[0]  # x_info
        r_s_b = rod_state[1]  # y_info
        r_s_c = rod_state[2]  # z_info

        num_points = int(self.n_elem / self.obs_state_points)
        # get full 3D state information
        rod_compact_state = np.concatenate((
            r_s_a[0:len(r_s_a) + 1:num_points],
            r_s_b[0:len(r_s_b) + 1:num_points],
            r_s_c[0:len(r_s_b) + 1:num_points],
        ))

        rod_compact_velocity = self.shearable_rod.velocity_collection[..., -1]
        rod_compact_velocity_norm = np.array([
            np.linalg.norm(rod_compact_velocity)
        ])
        rod_compact_velocity_dir = np.where(
            rod_compact_velocity_norm != 0,
            rod_compact_velocity / rod_compact_velocity_norm,
            0.0,
        )

        sphere_compact_state = self.target_point.flatten()  # 2
        sphere_compact_velocity = np.array([0, 0, 0])
        sphere_compact_velocity_norm = np.array([
            np.linalg.norm(sphere_compact_velocity)
        ])
        sphere_compact_velocity_dir = np.where(
            sphere_compact_velocity_norm != 0,
            sphere_compact_velocity / sphere_compact_velocity_norm,
            0.0,
        )
        rotate_matrix = compute_rotation_matrix(self.target_angle)
        self.target_tip_orientation = compute_quaternion_from_matrix(
            rotate_matrix
        )
        rotate_matrix = self.shearable_rod.director_collection[..., -1]
        self.rod_tip_orientation = compute_quaternion_from_matrix(
            rotate_matrix
        )

        state = np.concatenate((
            # rod information
            rod_compact_state,
            rod_compact_velocity_norm,
            rod_compact_velocity_dir,
            self.rod_tip_orientation,
            # target information
            sphere_compact_state,
            sphere_compact_velocity_norm,
            sphere_compact_velocity_dir,
            self.target_tip_orientation,
        ))

        return state

    def step(self, action):

        # action contains the control points for actuation torques in different directions in range [-1, 1]
        self.action = action

        self.spline_points_func_array_normal_dir[:
                                                 ] = action[:self.
                                                            number_of_control_points
                                                            ]
        self.spline_points_func_array_binormal_dir[:] = action[
            self.number_of_control_points:2 * self.number_of_control_points]
        self.spline_points_func_array_twist_dir[:] = action[
            2 * self.number_of_control_points:]

        self.current_step += 1

        # Do multiple time step of simulation for <one learning step>
        for _ in range(self.update_interval):
            self._do_step()

        # observe current state: current as sensed signal
        state = self.get_state()

        dist = np.linalg.norm(
            self.shearable_rod.position_collection[..., -1] - self.target_point
        )
        """ Reward Engineering """
        reward_dist = -np.square(dist).sum()

        ## distance between orientations from https://math.stackexchange.com/questions/90081/quaternion-distance
        # orientation_dist = (
        #     1.0 - np.dot(self.rod_tip_orientation, self.target_tip_orientation) ** 2
        # )
        # orientation_penalty = -((orientation_dist) ** 2)

        reward = 1.0 * reward_dist  # + 0.5 * orientation_penalty
        """ Done is a boolean to reset the environment before episode is completed """
        done = False

        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = isnan_check(
            self.shearable_rod.position_collection
        )

        if invalid_values_condition == True:
            print(" Nan detected in the position, exiting simulation now")
            self.shearable_rod.position_collection = np.zeros(
                self.shearable_rod.position_collection.shape
            )
            reward = -10000
            state = self.get_state()
            done = True

        if np.isclose(dist, 0.0, atol=0.05 * 2.0).all():
            reward += 0.5

        # for this specific case, check on_goal parameter
        if np.isclose(dist, 0.0, atol=0.05).all():
            reward += 1.5

        self.previous_action = action

        invalid_values_condition_state = isnan_check(state)
        if invalid_values_condition_state == True:
            print(
                " Nan detected in the state other than position data, exiting simulation now"
            )
            reward = -10000
            state = np.zeros(state.shape)
            done = True

        return state, reward, done, {"ctime": self.time_tracker}

    def is_achieve(self, eps: float = 0.02) -> bool:
        """
        Check if the rod has reached the target point.

        Parameters
        ----------
        eps : float
            Tolerance for checking if the rod has reached the target point.

        Returns
        -------
        bool
            True if the rod has reached the target point, False otherwise.
        """

        distance = np.linalg.norm(
            self.shearable_rod.position_collection[:, -1] - self.target_point
        )
        # print(distance)
        return distance < eps

    def render(self, mode="human"):
        """
        This method does nothing, it is here for interfacing with OpenAI Gym.

        Parameters
        ----------
        mode

        Returns
        -------

        """
        return
