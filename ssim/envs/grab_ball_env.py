__all__ = ["GrabBallEnvironment", "GrabBallArguments"]

from dataclasses import dataclass

import elastica as ea
import numpy as np
from elastica._calculus import _isnan_check

from ..arguments import (RodArguments, SimulatorArguments, SphereArguments,
                         SuperArguments)
from ..components import (ChangeableUniformForce,
                          RigidBodyAnalyticalLinearDamper)
from ..components.contact import JoinableRodSphereContact
from .base_envs import FetchableRodObjectsEnvironment


@dataclass
class GrabBallArguments(SuperArguments):

    rod: RodArguments
    objects: list[SphereArguments]
    simulator: SimulatorArguments


class GrabBallEnvironment(FetchableRodObjectsEnvironment):

    def __init__(self, configs: GrabBallArguments):
        self.rod_config = configs.rod
        self.object_configs = configs.objects
        self.sim_config = configs.simulator

        super().__init__(
            final_time=self.sim_config.final_time,
            time_step=self.sim_config.time_step,
            update_interval=self.sim_config.update_interval,
            rendering_fps=self.sim_config.rendering_fps,
        )

        self.uniform_force = np.array([0, 0, 1])

    def setup(self):

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

        for object_config in self.object_configs:
            if isinstance(object_config, SphereArguments):
                self.add_sphere(
                    center=object_config.center,
                    radius=object_config.radius,
                    density=object_config.density,
                )

        self.simulator.add_forcing_to(self.shearable_rod).using(
            ChangeableUniformForce,
            directional_force=self.uniform_force,
        )

        for obj in self.objects:
            self.simulator.detect_contact_between(
                self.shearable_rod,
                obj).using(JoinableRodSphereContact,
                           k=10,
                           nu=0,
                           velocity_damping_coefficient=1e3,
                           friction_coefficient=10,
                           flag=self.action_flags,
                           flag_id=self.object2id[obj])

        # for i in range(len(self.objects)):
        #     for j in range(i + 1, len(self.objects)):
        #         self.simulator.detect_contact_between(
        #             self.objects[i], self.objects[j]).using(
        #                 ea.SphereSphereContact,
        #                 k=10,
        #                 nu=0,
        #                 velocity_damping_coefficient=1e3,
        #                 friction_coefficient=10)

        damping_constant = 2e-2
        self.simulator.dampen(self.shearable_rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )
        for idx, object_ in enumerate(self.objects):
            dampen = 10000 if idx == 1 else 1
            self.simulator.dampen(object_).using(
                RigidBodyAnalyticalLinearDamper,
                damping_constant=dampen,
                time_step=self.time_step,
            )

        callback_step_skip = int(
            1.0 / (self.sim_config.rendering_fps * self.time_step))
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
