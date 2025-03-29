__all__ = ["GrabMeshEnvironment", "GrabMeshArguments"]

from dataclasses import dataclass
from typing import Optional, Sequence

import elastica as ea
import numpy as np
from elastica import PositionVerlet
from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface

from ..components.contact import JoinableRodSphereContact
from ..components.surface import MeshSurface
from ..arguments import (RodArguments, SimulatorArguments, SphereArguments,
                         SuperArguments)
from ..components import (ChangeableUniformForce,
                          RigidBodyAnalyticalLinearDamper,
                          RodMeshSurfaceContactWithGridMethod)
from .base_envs import RodObjectsEnvironment


@dataclass
class GrabMeshArguments(SuperArguments):

    rod: RodArguments
    objects: Sequence[SphereArguments] = ()
    simulator: SimulatorArguments = None


class GrabMeshEnvironment(RodObjectsEnvironment):

    def __init__(self, configs: GrabMeshArguments):
        super().__init__()
        self.rod_config = configs.rod
        self.object_configs = configs.objects
        self.sim_config = configs.simulator

        self.stateful_stepper = PositionVerlet()

        self.uniform_force = np.array([0, 0, 1])

        self.total_steps = int(self.sim_config.final_time /
                               self.sim_config.time_step)
        self.time_step = np.float64(
            float(self.sim_config.final_time) / self.total_steps)
        self.rendering_fps = self.sim_config.rendering_fps
        self.time_tracker = np.float64(0.0)

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

        self.cube = MeshSurface("/data/zyw/workshop/PyElastica/tests/cube.stl")
        self.cube.translate(np.array([0, 0, 2]))
        self.simulator.append(self.cube)
        from stl import mesh
        mesh_data = mesh.Mesh.from_file(
            "/data/zyw/workshop/PyElastica/tests/cube.stl")
        # self.cube.visualize()
        from ssim.components.contact import surface_grid
        grid_size = 0.1  # 网格大小
        faces_grid = surface_grid(mesh_data.vectors, grid_size)
        faces_grid["model_path"] = self.cube.model_path
        faces_grid["grid_size"] = grid_size
        faces_grid["surface_reorient"] = self.cube.mesh_orientation
        k = 1e4  # 接触刚度系数
        nu = 10  # 阻尼系数
        surface_tol = 1e-2
        self.simulator.detect_contact_between(
            self.shearable_rod,
            self.cube).using(RodMeshSurfaceContactWithGridMethod,
                             k=k,
                             nu=nu,
                             faces_grid=faces_grid,
                             grid_size=grid_size,
                             surface_tol=surface_tol)

        callback_step_skip = int(
            1.0 / (self.sim_config.rendering_fps * self.time_step))
        # Add callbacks for data collection if enabled
        self._add_data_collection_callbacks(callback_step_skip)

        # Finalize the simulator
        self.simulator.finalize()

        # Prepare for time stepping
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.stateful_stepper, self.simulator)

        return self

    def step(self, num_steps: int = 1):
        for _ in range(num_steps):
            self.time_tracker = self.do_step(
                self.stateful_stepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )

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
