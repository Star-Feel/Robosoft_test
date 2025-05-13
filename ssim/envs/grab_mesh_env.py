__all__ = [
    "MeshDemoArguments",
    "MeshDemoEnvironment",
]

from dataclasses import dataclass
from typing import Sequence

import elastica as ea
import numpy as np
from elastica import RigidBodyBase
from elastica._calculus import _isnan_check
from stl import mesh

from ..arguments import (MeshSurfaceArguments, RodArguments,
                         SimulatorArguments, SphereArguments, SuperArguments)
from ..components import (ChangeableUniformForce,
                          RigidBodyAnalyticalLinearDamper,
                          RodMeshSurfaceContactWithGridMethod)
from ..components.contact import JoinableRodSphereContact, surface_grid
from ..components.surface.mesh_surface import MeshSurface
from .base_envs import FetchableRodObjectsEnvironment


@dataclass
class MeshDemoArguments(SuperArguments):

    rod: RodArguments
    objects: Sequence[SphereArguments | MeshSurfaceArguments] = ()
    simulator: SimulatorArguments = None


class MeshDemoEnvironment(FetchableRodObjectsEnvironment):

    def __init__(self, configs: MeshDemoArguments):

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
            elif isinstance(object_config, MeshSurfaceArguments):
                self.add_mesh_surface(object_config.mesh_path,
                                      object_config.center,
                                      object_config.scale,
                                      object_config.rotate)

        self.simulator.add_forcing_to(self.shearable_rod).using(
            ChangeableUniformForce,
            directional_force=self.uniform_force,
        )

        for obj in self.objects:
            if isinstance(obj, ea.Sphere):
                self.simulator.detect_contact_between(
                    self.shearable_rod,
                    obj).using(JoinableRodSphereContact,
                               k=10,
                               nu=0,
                               velocity_damping_coefficient=1e3,
                               friction_coefficient=10,
                               flag=self.action_flags,
                               flag_id=self.object2id[obj])
            elif isinstance(obj, MeshSurface):
                mesh_data = mesh.Mesh.from_file(obj.model_path)
                grid_size = 0.1  # 网格大小
                faces_grid = surface_grid(mesh_data.vectors, grid_size)
                faces_grid["model_path"] = obj.model_path
                faces_grid["grid_size"] = grid_size
                faces_grid["surface_reorient"] = obj.mesh_orientation
                k = 1e4  # 接触刚度系数
                nu = 10  # 阻尼系数
                surface_tol = 1e-2
                self.simulator.detect_contact_between(
                    self.shearable_rod,
                    obj).using(RodMeshSurfaceContactWithGridMethod,
                               k=k,
                               nu=nu,
                               faces_grid=faces_grid,
                               grid_size=grid_size,
                               surface_tol=surface_tol)

        damping_constant = 2e-2
        self.simulator.dampen(self.shearable_rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )
        for _, object_ in enumerate(self.objects):
            if isinstance(object_, RigidBodyBase):
                dampen = 1
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
