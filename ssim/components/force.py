__all__ = ["ChangeableUniformForce"]

import numpy as np
from elastica import NoForces
from elastica.typing import RodType


class ChangeableUniformForce(NoForces):

    def __init__(self, directional_force=np.array([0.0, 0.0, 0.0])):
        self.force = directional_force

    def apply_forces(self, rod: RodType, time: np.float64 = 0.0):
        force_on_one_element = (self.force / rod.n_elems).reshape(3, 1)

        rod.external_forces += force_on_one_element

        # Because mass of first and last node is half
        rod.external_forces[..., 0] -= 0.5 * force_on_one_element[:, 0]
        rod.external_forces[..., -1] -= 0.5 * force_on_one_element[:, 0]
