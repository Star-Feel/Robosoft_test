from typing import Optional

import elastica as ea
import numpy as np
from elastica import (BaseSystemCollection, CallBacks, Connections,
                      Constraints, Contact, Damping, Forcing, PositionVerlet)
from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface

from ..arguments import RodArguments, SimulatorArguments, SphereArguments
from ..components import RigidBodyAnalyticalLinearDamper
from ..visualize.visualizer import create_3d_animation, plot_video


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing,
                    Damping, Contact, CallBacks):
    """Base simulator class combining Elastica functionalities."""
    pass


class ContinuumSnakeCallBack(ea.CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(
                system.position_collection.copy())
            self.callback_params["velocity"].append(
                system.velocity_collection.copy())
            self.callback_params["avg_velocity"].append(
                system.compute_velocity_center_of_mass())
            self.callback_params["center_of_mass"].append(
                system.compute_position_center_of_mass())
            self.callback_params["curvature"].append(system.kappa.copy())

            return


class CylinderCallBack(ea.CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(
                system.position_collection.copy())
            self.callback_params["velocity"].append(
                system.velocity_collection.copy())

            self.callback_params["center_of_mass"].append(
                system.compute_position_center_of_mass())

            return


class PushBallEnvironment:

    def __init__(
        self,
        rod_config: RodArguments,
        sphere_config: SphereArguments,
        sim_config: SimulatorArguments,
    ):
        self.rod_config = rod_config
        self.sphere_config = sphere_config
        self.sim_config = sim_config
        self.stateful_stepper = PositionVerlet()

        self.total_steps = int(self.sim_config.final_time /
                               self.sim_config.time_step)
        self.time_step = np.float64(
            float(self.sim_config.final_time) / self.total_steps)
        self.rendering_fps = self.sim_config.rendering_fps
        self.time_tracker = np.float64(0.0)

        self.sphere_callback_data = ea.defaultdict(list)
        self.rod_callback_data = ea.defaultdict(list)

    def setup(self):

        self.simulator = BaseSimulator()

        shear_modulus = self.rod_config.youngs_modulus / (
            self.rod_config.poisson_ratio + 1.0)

        self.shearable_rod = ea.CosseratRod.straight_rod(
            self.rod_config.n_elem,
            self.rod_config.start,
            self.rod_config.direction,
            self.rod_config.normal,
            self.rod_config.base_length,
            self.rod_config.base_radius,
            self.rod_config.density,
            youngs_modulus=self.rod_config.youngs_modulus,
            shear_modulus=shear_modulus,
        )

        # Append rod to the simulator
        self.simulator.append(self.shearable_rod)

        self.sphere = ea.Sphere(self.sphere_config.center,
                                self.sphere_config.radius,
                                density=self.sphere_config.density)
        self.simulator.append(self.sphere)

        force_magnitude = 0.2
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ea.UniformForces,
            force=force_magnitude,
            direction=np.array([0.0, 0.0, 1.0]),
        )

        self.simulator.detect_contact_between(
            self.shearable_rod,
            self.sphere).using(ea.RodSphereContact,
                               k=10,
                               nu=0,
                               velocity_damping_coefficient=1e3,
                               friction_coefficient=10)

        damping_constant = 2e-2
        self.simulator.dampen(self.shearable_rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )
        self.simulator.dampen(self.sphere).using(
            RigidBodyAnalyticalLinearDamper,
            damping_constant=1,
            time_step=self.time_step,
        )

        # Add callbacks for data collection if enabled
        self._add_data_collection_callbacks()

        # Finalize the simulator
        self.simulator.finalize()

        # Prepare for time stepping
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.stateful_stepper, self.simulator)

        return self

    def _add_data_collection_callbacks(self):
        callback_step_skip = int(
            1.0 / (self.sim_config.rendering_fps * self.time_step))
        self.simulator.collect_diagnostics(self.shearable_rod).using(
            ContinuumSnakeCallBack,
            step_skip=callback_step_skip,
            callback_params=self.rod_callback_data)
        self.simulator.collect_diagnostics(self.sphere).using(
            CylinderCallBack,
            step_skip=callback_step_skip,
            callback_params=self.sphere_callback_data)

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

    def visualize_3d(self, save_path: Optional[str] = None):
        create_3d_animation(np.array(self.rod_callback_data["position"]),
                            np.array(self.sphere_callback_data["position"]),
                            self.sphere.radius,
                            save_path=save_path,
                            fps=self.sim_config.rendering_fps)

    def visualize_2d(self, save_path: Optional[str] = None):
        plot_video(
            self.rod_callback_data,
            self.sphere_callback_data,
            self.sphere,
            save_path,
            fps=self.sim_config.rendering_fps,
            xlim=(0, 4),
            ylim=(-1, 1),
        )
