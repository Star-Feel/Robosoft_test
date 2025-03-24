__all__ = ["JoinableRodSphereContact"]

from elastica import RodSphereContact, RodType, AllowedContactType


class JoinableRodSphereContact(RodSphereContact):

    def __init__(
        self,
        k: float,
        nu: float,
        velocity_damping_coefficient=0.0,
        friction_coefficient=0.0,
        index: int = -1,
        flag: list[bool] = [False],
        flag_id: int = 0,
    ):
        """
        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        velocity_damping_coefficient : float
            Velocity damping coefficient between rigid-body and rod contact is used to apply friction force in the
            slip direction.
        friction_coefficient : float
            For Coulombic friction coefficient for rigid-body and rod contact.
        """
        super().__init__(
            k,
            nu,
            velocity_damping_coefficient,
            friction_coefficient,
        )

        self.index = index
        self.flag = flag
        self.flag_id = flag_id
        self.relative_position = None

    def _check_systems_validity(
        self,
        system_one,
        system_two,
    ) -> None:
        pass

    def apply_contact(self, system_one: RodType,
                      system_two: AllowedContactType) -> None:
        """
        Apply contact forces and torques between RodType object and Sphere object.

        Parameters
        ----------
        system_one: object
            Rod object.
        system_two: object
            Sphere object.

        """
        if not self.flag[self.flag_id]:
            self.relative_position = None
            super().apply_contact(system_one, system_two)
        else:
            if self.relative_position is None:
                self.relative_position = \
                    system_two.position_collection[..., 0] - \
                    system_one.position_collection[..., self.index]
            else:
                system_two.position_collection[..., 0] = \
                    system_one.position_collection[..., self.index] \
                    + self.relative_position

            system_two.velocity_collection[
                ..., 0] = system_one.velocity_collection[..., self.index]
