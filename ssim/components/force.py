__all__ = [
    "ChangeableUniformForce",
    "ChangeableMuscleTorques",
]

from typing import Optional
import numpy as np
from elastica import NoForces
from elastica.typing import RodType
from elastica import MuscleTorques
from elastica.external_forces import inplace_addition, inplace_substraction
from elastica._linalg import _batch_matvec

from numba import njit
from elastica._linalg import _batch_product_i_k_to_ik


class ChangeableUniformForce(NoForces):

    def __init__(self, directional_force=np.array([0.0, 0.0, 0.0])):
        self.force = directional_force

    def apply_forces(self, rod: RodType, time: np.float64 = 0.0):
        force_on_one_element = (self.force / rod.n_elems).reshape(3, 1)

        rod.external_forces += force_on_one_element

        # Because mass of first and last node is half
        rod.external_forces[..., 0] -= 0.5 * force_on_one_element[:, 0]
        rod.external_forces[..., -1] -= 0.5 * force_on_one_element[:, 0]


class ChangeableMuscleTorques(MuscleTorques):

    DIRECT = 0
    LEFT = 1
    RIGHT = 2

    def __init__(self,
                 *args,
                 turn: list[int],
                 callbacks: Optional[list] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.amplitude = 1.0
        self.turn = turn
        self.callbacks = callbacks
        self.turn_start_time = None
        self.causal_mask = np.ones_like(self.my_spline)
        self.phase_shift = np.zeros_like(self.my_spline)

    def update_amplitude(self, s, time, angular_frequency, wave_number,
                         phase_shift, amplitude_factor):
        sign = np.sign(
            np.sin(angular_frequency * time - wave_number * s + phase_shift))
        self.amplitude = 1 + sign * amplitude_factor

    def update_phase_shift(self, time, turn: int = 0):
        if turn == self.DIRECT:
            self.turn_start_time = None
            self.phase_shift = np.zeros_like(self.my_spline)
            return
        elif turn == self.LEFT:
            direction = -1
        elif turn == self.RIGHT:
            direction = 1
        else:
            raise ValueError("Invalid turn value. Use 0, 1, or 2.")

        if self.turn_start_time is None:
            self.turn_start_time = time

        sign = np.sign(
            np.sin(self.angular_frequency * time - self.wave_number * self.s +
                   self.phase_shift))
        causal_mask = time > (
            self.turn_start_time +
            (self.s - self.s[0]) * self.wave_number / self.angular_frequency)
        self.causal_mask = causal_mask.astype(int)
        self.phase_shift = direction * self.causal_mask * sign * np.pi / 6

    def apply_torques(self, rod, time):
        self.update_phase_shift(time, self.turn[0])

        torque = self.compute_muscle_torques(
            time,
            self.my_spline,
            self.s,
            self.angular_frequency,
            self.wave_number,
            self.phase_shift,
            self.ramp_up_time,
            self.direction,
            rod.director_collection,
            rod.external_torques,
            self.amplitude,
        )
        if self.callbacks is not None:
            self.callbacks.append(torque)

    @staticmethod
    @njit(cache=True)
    def compute_muscle_torques(
        time,
        my_spline,
        s,
        angular_frequency,
        wave_number,
        phase_shift,
        ramp_up_time,
        direction,
        director_collection,
        external_torques,
        amplitude,
    ):
        # Ramp up the muscle torque
        factor = min(1.0, time / ramp_up_time)
        # From the node 1 to node nelem-1
        # Magnitude of the torque. Am = beta(s) * sin(2pi*t/T + 2pi*s/lambda + phi)
        # There is an inconsistency with paper and Elastica cpp implementation. In paper sign in
        # front of wave number is positive, in Elastica cpp it is negative.
        torque_mag = (
            factor * my_spline * amplitude *
            np.sin(angular_frequency * time - wave_number * s + phase_shift))
        # Head and tail of the snake is opposite compared to elastica cpp. We need to iterate torque_mag
        # from last to first element.
        torque = _batch_product_i_k_to_ik(direction, torque_mag[::-1])

        inplace_addition(
            external_torques[..., 1:],
            _batch_matvec(director_collection, torque)[..., 1:],
        )
        inplace_substraction(
            external_torques[..., :-1],
            _batch_matvec(director_collection[..., :-1], torque[..., 1:]),
        )
        return torque
