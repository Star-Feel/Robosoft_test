import numpy as np

from elastica import MuscleTorques
from elastica.external_forces import inplace_addition, inplace_substraction
from elastica._linalg import _batch_matvec

from numba import njit
from elastica._linalg import _batch_product_i_k_to_ik


class ChangeableMuscleTorques(MuscleTorques):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amplitude = 1.0
        self.causal_mask = np.ones_like(self.my_spline)
        self.phase_shift = np.zeros_like(self.my_spline)

    def update_amplitude(self, s, time, angular_frequency, wave_number, phase_shift, amplitude_factor):
        sign = np.sign(np.sin(angular_frequency * time - wave_number * s + phase_shift))
        self.amplitude = 1 + sign * amplitude_factor
    
    def update_phase_shift(self, time, start_time=2.5, end_time=7.5, turn_left:str=False):
        sign = np.sign(np.sin(self.angular_frequency * time - self.wave_number * self.s + self.phase_shift))
        if time < start_time:
            self.phase_shift = np.zeros_like(self.my_spline)
            return
        causal_mask = time > (start_time + (self.s-self.s[0]) * self.wave_number / self.angular_frequency)
        self.causal_mask = causal_mask.astype(int)
        direction = -1 if turn_left else 1
        if time > end_time:
            self.phase_shift = np.zeros_like(self.my_spline)
        else:
            self.phase_shift =  direction * self.causal_mask * sign * np.pi/6

    def apply_torques(self, rod, time):
        self.update_phase_shift(time, start_time=2.5, end_time=5, turn_left=False)

        self.compute_muscle_torques(
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
            factor
            * my_spline
            * amplitude
            * np.sin(angular_frequency * time - wave_number * s + phase_shift)
        )
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