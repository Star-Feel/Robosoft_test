from elastica import ConstraintBase


class PinJoint(ConstraintBase):

    def __init__(self, other, index, flag=[False], **kwargs):
        super().__init__(**kwargs)
        self.other = other
        self.index = index
        self.flag = flag

    def constrain_values(self, system, time: float) -> None:
        if self.flag[0]:
            system.position_collection[
                ..., 0] = self.other.position_collection[..., self.index]
        # system.director_collection[..., 0] = self.other.position_collection[
        # ..., self.index]

    def constrain_rates(self, system, time: float) -> None:
        # system.velocity_collection[..., 0] = 0.0
        # system.omega_collection[..., 0] = 0.0
        pass
