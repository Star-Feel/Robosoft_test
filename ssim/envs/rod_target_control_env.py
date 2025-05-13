__all__ = [
    "SoftTargetControlArguments",
    "SoftTargetControlEnvironment",
]

from collections import defaultdict
from dataclasses import dataclass

import elastica as ea
import numpy as np
from elastica import OneEndFixedRod

from ..arguments import (RodArguments, RodControllerArgumets,
                         SimulatorArguments, SphereArguments, SuperArguments)
from ..components import (MuscleTorquesWithVaryingBetaSplines,
                          RigidBodyAnalyticalLinearDamper)
from ..utils import (compute_quaternion_from_matrix, compute_rotation_matrix,
                     isnan_check)
from .base_envs import RodControlMixin, RodSphereEnvironment, SimulatedEnvironment


@dataclass
class SoftTargetControlArguments(SuperArguments):

    rod: RodArguments
    sphere: SphereArguments
    simulator: SimulatorArguments = None
    controller: RodControllerArgumets = None


class SoftTargetControlEnvironment(RodSphereEnvironment, RodControlMixin,
                                   SimulatedEnvironment):

    def __init__(self, configs: SoftTargetControlArguments):

        self.rod_config = configs.rod
        self.sphere_config = configs.sphere
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

    def setup(self):

        shear_modulus = self.rod_config.youngs_modulus / (
            self.rod_config.poisson_ratio + 1.0)

        radius_tip = 0.05
        radius_along_rod = np.linspace(self.rod_config.base_radius, radius_tip,
                                       self.rod_config.n_elem)
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

        if self.trainable:
            t_x = np.random.uniform(self.boundary[0], self.boundary[1])
            t_y = np.random.uniform(self.boundary[2], self.boundary[3])
            t_z = np.random.uniform(self.boundary[4], self.boundary[5])
            target_position = np.array([t_x, t_y, t_z])
            theta_x = 0
            theta_y = np.random.uniform(np.pi / 6, np.pi / 3)
            theta_z = 0
            theta = np.array([theta_x, theta_y, theta_z])
        else:
            target_position = self.sphere_config.center
            theta = self.sphere_config.direction / 180 * np.pi

        print("Target position:", target_position)
        print("Target theta:", theta)
        self.sphere = self.add_sphere(
            center=target_position,
            radius=self.sphere_config.radius,
            density=self.sphere_config.density,
            theta=theta,
        )
        rotate_matrix = compute_rotation_matrix(theta)
        self.sphere.director_collection[..., 0] = rotate_matrix

        quart = self.sphere.director_collection[..., 0]
        self.target_tip_orientation = compute_quaternion_from_matrix(quart)

        # fix one end of the rod
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod,
            constrained_position_idx=(0, ),
            constrained_director_idx=(0, ))
        # add callbacks
        callback_step_skip = int(
            1.0 / (self.sim_config.rendering_fps * self.time_step))
        # Add muscle torques acting on the arm for actuation
        # MuscleTorquesWithVaryingBetaSplines uses the control points selected
        # by RL to generate torques along the arm.
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        self.spline_points_func_array_normal_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.rod_config.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_normal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("normal"),
            step_skip=callback_step_skip,
            max_rate_of_change_of_activation=np.infty,
            torque_profile_recorder=self.
            torque_profile_list_for_muscle_in_normal_dir,
        )

        self.torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)
        self.spline_points_func_array_binormal_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.rod_config.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_binormal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("binormal"),
            step_skip=callback_step_skip,
            max_rate_of_change_of_activation=np.infty,
            torque_profile_recorder=self.
            torque_profile_list_for_muscle_in_binormal_dir,
        )

        self.torque_profile_list_for_muscle_in_twist_dir = defaultdict(list)
        self.spline_points_func_array_twist_dir = []
        # Apply torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.rod_config.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_twist_dir,
            muscle_torque_scale=self.beta,
            direction=str("tangent"),
            step_skip=callback_step_skip,
            max_rate_of_change_of_activation=np.infty,
            torque_profile_recorder=self.
            torque_profile_list_for_muscle_in_twist_dir,
        )

        # damping of rod and objects
        damping_constant = 10
        self.simulator.dampen(self.shearable_rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )

        dampen = 10000
        self.simulator.dampen(self.sphere).using(
            RigidBodyAnalyticalLinearDamper,
            damping_constant=dampen,
            time_step=self.time_step,
        )

        # Add callbacks for data collection if enabled
        self._add_data_collection_callbacks(callback_step_skip)

        self._finalize()

        if self.trainable:
            # reset on_goal
            self.on_goal = 0
            # reset current_step
            self.current_step = 0
            # reset time_tracker
            self.time_tracker = np.float64(0.0)
            # reset previous_action
            self.previous_action = None

        return self.get_state()

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
        rod_compact_velocity_norm = np.array(
            [np.linalg.norm(rod_compact_velocity)])
        rod_compact_velocity_dir = np.where(
            rod_compact_velocity_norm != 0,
            rod_compact_velocity / rod_compact_velocity_norm,
            0.0,
        )

        sphere_compact_state = self.sphere.position_collection.flatten()  # 2
        sphere_compact_velocity = self.sphere.velocity_collection.flatten()
        sphere_compact_velocity_norm = np.array(
            [np.linalg.norm(sphere_compact_velocity)])
        sphere_compact_velocity_dir = np.where(
            sphere_compact_velocity_norm != 0,
            sphere_compact_velocity / sphere_compact_velocity_norm,
            0.0,
        )

        rotate_matrix = self.shearable_rod.director_collection[..., -1]
        self.rod_tip_orientation = compute_quaternion_from_matrix(
            rotate_matrix)

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

        self.spline_points_func_array_normal_dir[:] = \
            action[:self.number_of_control_points]
        self.spline_points_func_array_binormal_dir[:] = action[
            self.number_of_control_points:2 * self.number_of_control_points]
        self.spline_points_func_array_twist_dir[:] = action[
            2 * self.number_of_control_points:]

        # Do multiple time step of simulation for <one learning step>
        for _ in range(self.update_interval):
            self._do_step()

        # observe current state: current as sensed signal
        state = self.get_state()

        self.previous_action = action

        if self.trainable:
            self.current_step += 1

            # observe current state: current as sensed signal
            state = self.get_state()

            dist = np.linalg.norm(self.shearable_rod.position_collection[...,
                                                                         -1] -
                                  self.sphere.position_collection[..., 0])
            """ Reward Engineering """
            reward_dist = -np.square(dist).sum()

            ## distance between orientations from https://math.stackexchange.com/questions/90081/quaternion-distance
            orientation_dist = (1.0 - np.dot(self.rod_tip_orientation,
                                             self.target_tip_orientation)**2)
            orientation_penalty = -((orientation_dist)**2)

            reward = 1.0 * reward_dist + 0.5 * orientation_penalty
            """ Done is a boolean to reset the environment before episode is completed """
            done = False

            # Position of the rod cannot be NaN, it is not valid, stop the simulation
            invalid_values_condition = isnan_check(
                self.shearable_rod.position_collection)

            if invalid_values_condition == True:
                print(" Nan detected in the position, exiting simulation now")
                self.shearable_rod.position_collection = np.zeros(
                    self.shearable_rod.position_collection.shape)
                reward = -10000
                state = self.get_state()
                done = True

            if np.isclose(dist, 0.0, atol=0.05 * 2.0).all():
                reward += 0.5
                reward += 0.5 * (1 - orientation_dist)
                if np.isclose(orientation_dist, 0.0, atol=0.05 * 2.0).all():
                    reward += 0.5

            # for this specific case, check on_goal parameter
            if np.isclose(dist, 0.0, atol=0.05).all():
                reward += 1.5
                reward += 1.5 * (1 - orientation_dist)
                if np.isclose(orientation_dist, 0.0, atol=0.05).all():
                    reward += 1.5

            if self.current_step >= self.total_learning_steps:
                done = True
                if reward > 0:
                    print(
                        " Reward greater than 0! Reward: %0.3f, Distance: %0.3f, Orientation: %0.3f -- %0.3f, %0.3f "
                        % (reward, dist, orientation_dist, reward_dist,
                           orientation_penalty))
                else:
                    print(
                        " Finished simulation. Reward: %0.3f, Distance: %0.3f, Orientation: %0.3f -- %0.3f, %0.3f"
                        % (reward, dist, orientation_dist, reward_dist,
                           orientation_penalty))
            """ Done is a boolean to reset the environment before episode is completed """

            self.previous_action = action

            invalid_values_condition_state = isnan_check(state)
            if invalid_values_condition_state == True:
                print(
                    " Nan detected in the state other than position data, exiting simulation now"
                )
                reward = -10000
                state = np.zeros(state.shape)
                done = True
            if state.shape[0] != 52:
                print()
            return state, reward, done, {"ctime": self.time_tracker}

        return state
