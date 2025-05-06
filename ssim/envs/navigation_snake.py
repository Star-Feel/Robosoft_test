__all__ = ["NavigationSnakeEnvironment", "NavigationSnakeArguments"]

from dataclasses import dataclass

import elastica as ea
import numpy as np
from elastica._calculus import _isnan_check

from ..arguments import (RodArguments, SimulatorArguments, SphereArguments,
                         SuperArguments)
from .base_envs import RodObjectsEnvironment

from ..components import ChangeableMuscleTorques


@dataclass
class NavigationSnakeArguments(SuperArguments):

    rod: RodArguments
    objects: list[SphereArguments]
    simulator: SimulatorArguments


class NavigationSnakeEnvironment(RodObjectsEnvironment):

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
            self.rod_config.poisson_ratio + 1.0)

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
        period = 1.0

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
        mu = self.rod_config.base_length / (period * period *
                                            np.abs(gravitational_acc) * froude)
        kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu
                                     ])  # [forward, backward, sideways]
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
            1.0 /
            (self.sim_config.rendering_fps *
             self.time_step)) if callback_step_skip < 0 else callback_step_skip
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
            "time":
            self.time_tracker,
            "rod_tip_position":
            self.shearable_rod.position_collection[..., -1].copy(),
            "status":
            "ok"
        }

    def set_target(self, target: list[float]):
        self.target = np.array(target)

    def reach(self, eps=0.01):
        rod_tip_position = self.shearable_rod.position_collection[...,
                                                                  -1].copy()
        distance = np.linalg.norm(rod_tip_position - self.target)
        return distance < eps
