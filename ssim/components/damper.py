__all__ = ["RigidBodyAnalyticalLinearDamper"]

import numpy as np
from elastica import DamperBase
from elastica.rigidbody.rigid_body import RigidBodyBase


class RigidBodyAnalyticalLinearDamper(DamperBase):

    def __init__(self, damping_constant, time_step, **kwargs):
        """
        Analytical linear damper initializer

        Parameters
        ----------
        damping_constant : float
            Damping constant for the analytical linear damper.
        time_step : float
            Time-step of simulation
        """
        super().__init__(**kwargs)
        # Compute the damping coefficient for translational velocity
        mass = self._system.mass
        self.translational_damping_coefficient = np.exp(-damping_constant *
                                                        time_step)

        # Compute the damping coefficient for exponential velocity
        self.rotational_damping_coefficient = np.exp(
            -damping_constant * time_step * mass *
            np.diagonal(self._system.inv_mass_second_moment_of_inertia).T)

    def dampen_rates(self, system: RigidBodyBase, time: float):
        system.velocity_collection[:] = (
            system.velocity_collection *
            self.translational_damping_coefficient)

        system.omega_collection[:] = system.omega_collection * np.power(
            self.rotational_damping_coefficient, 1.0)
