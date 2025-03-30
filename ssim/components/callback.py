__all__ = [
    "RodCallBack",
    "RigidBodyCallBack",
    "MeshSurfaceCallBack",
]

import elastica as ea


class RodCallBack(ea.CallBackBaseClass):
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
            self.callback_params["radius"].append(system.radius.copy())


class RigidBodyCallBack(ea.CallBackBaseClass):
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
            self.callback_params["radius"].append(system.radius.copy())


class MeshSurfaceCallBack(ea.CallBackBaseClass):
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
            self.callback_params["position"].append(system.mesh_center.copy())
            self.callback_params["orientation"].append(
                system.mesh_orientation.copy())
            self.callback_params["scale"].append(system.mesh_scale.copy())
            self.callback_params["model_path"].append(system.model_path)
            self.callback_params["face_centers"].append(
                system.face_centers.copy())
            self.callback_params["face_normals"].append(
                system.face_normals.copy())
            self.callback_params["faces"].append(system.faces.copy())
