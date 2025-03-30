__all__ = [
    "JoinableRodSphereContact",
    "RodMeshSurfaceContactWithGridMethod",
]
import elastica
from elastica import NoContact, RodSphereContact, RodType, AllowedContactType, SystemType
from elastica.interaction import node_to_element_position
from elastica.interaction import node_to_element_velocity, elements_to_nodes_inplace
from elastica._linalg import _batch_product_k_ik_to_ik, _batch_dot, _batch_norm
from elastica.rod import RodBase
from .surface import MeshSurface
import numba
import numpy as np


def find_contact_faces_idx(faces_grid, x_min, y_min, grid_size,
                           position_collection):
    element_position = elastica.contact_utils._node_to_element_position(
        position_collection)
    n_element = element_position.shape[-1]
    position_idx_array = np.empty((0))
    face_idx_array = np.empty((0))
    grid_position = np.round(
        (element_position[0:2, :] - np.array([x_min, y_min]).reshape(
            (2, 1))) / grid_size)

    # find face neighborhood of each element position

    for i in range(n_element):
        try:
            face_idx_1 = faces_grid[(int(grid_position[0, i]),
                                     int(grid_position[1,
                                                       i]))]  # first quadrant
        except Exception:
            face_idx_1 = np.empty((0))
        try:
            face_idx_2 = faces_grid[(int(grid_position[0, i] - 1),
                                     int(grid_position[1,
                                                       i]))]  # second quadrant
        except Exception:
            face_idx_2 = np.empty((0))
        try:
            face_idx_3 = faces_grid[(int(grid_position[0, i] - 1),
                                     int(grid_position[1, i] -
                                         1))]  # third quadrant
        except Exception:
            face_idx_3 = np.empty((0))
        try:
            face_idx_4 = faces_grid[(int(grid_position[0, i]),
                                     int(grid_position[1, i] -
                                         1))]  # fourth quadrant
        except Exception:
            face_idx_4 = np.empty((0))
        face_idx_element = np.concatenate(
            (face_idx_1, face_idx_2, face_idx_3, face_idx_4))
        face_idx_element_no_duplicates = np.unique(face_idx_element)
        if face_idx_element_no_duplicates.size == 0:
            raise RuntimeError(
                "Rod object out of grid bounds"
            )  # a rod element is on four grids with no faces

        face_idx_array = np.concatenate(
            (face_idx_array, face_idx_element_no_duplicates))
        n_contacts = face_idx_element_no_duplicates.shape[0]
        position_idx_array = np.concatenate((position_idx_array, i * np.ones(
            (n_contacts, ))))

    position_idx_array = position_idx_array.astype(int)
    face_idx_array = face_idx_array.astype(int)
    return position_idx_array, face_idx_array, element_position


@numba.njit(cache=True)
def surface_grid_numba(faces, grid_size, face_x_left, face_x_right,
                       face_y_down, face_y_up):
    """
    Computes the faces_grid dictionary for rod-meshsurface contact
    Consider calling surface_grid for face_grid generation
    """
    x_min = np.min(faces[0, :, :])
    y_min = np.min(faces[1, :, :])
    n_x_positions = int(np.ceil((np.max(faces[0, :, :]) - x_min) / grid_size))
    n_y_positions = int(np.ceil((np.max(faces[1, :, :]) - y_min) / grid_size))
    faces_grid = dict()
    for i in range(n_x_positions):
        x_left = x_min + (i * grid_size)
        x_right = x_min + ((i + 1) * grid_size)
        for j in range(n_y_positions):
            y_down = y_min + (j * grid_size)
            y_up = y_min + ((j + 1) * grid_size)
            if np.any(
                    np.where(((face_y_down > y_up) + (face_y_up < y_down) +
                              (face_x_right < x_left) +
                              (face_x_left > x_right)) == 0)[0]):
                faces_grid[(i,
                            j)] = np.where(((face_y_down > y_up) +
                                            (face_y_up < y_down) +
                                            (face_x_right < x_left) +
                                            (face_x_left > x_right)) == 0)[0]
    return faces_grid


def surface_grid(faces, grid_size):
    """
    Returns the faces_grid dictionary for rod-meshsurface contact
    This function only creates the faces_grid dict;
    the user must store the data in a binary file using pickle.dump
    """
    face_x_left = np.min(faces[0, :, :], axis=0)
    face_y_down = np.min(faces[1, :, :], axis=0)
    face_x_right = np.max(faces[0, :, :], axis=0)
    face_y_up = np.max(faces[1, :, :], axis=0)

    return dict(
        surface_grid_numba(faces, grid_size, face_x_left, face_x_right,
                           face_y_down, face_y_up))


@numba.njit(cache=True, nopython=True)
def _calculate_contact_forces_rod_mesh_surface(
    faces_normals: np.ndarray,
    faces_centers: np.ndarray,
    element_position: np.ndarray,
    position_idx_array: np.ndarray,
    face_idx_array,
    surface_tol: float,
    k: float,
    nu: float,
    radius: np.array,
    mass: np.array,
    velocity_collection: np.ndarray,
    external_forces: np.ndarray,
) -> tuple:
    """
    This function computes the plane force response on the element, in the
    case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
    is used.

    Parameters
    ----------
    faces_normals: np.ndarray
        mesh cell's normal vectors
    faces_centers: np.ndarray
        mesh cell's center points
    element_position: np.ndarray
        rod element's center points
    position_idx_array: np.ndarray
        rod element's index array
    face_idx_array: np.ndarray
        mesh cell's index array
    surface_tol: float
        Penetration tolerance between the surface and the rod-like object
    k: float
        Contact spring constant
    nu: float
        Contact damping constant
    radius: np.array
        rod element's radius
    mass: np.array
        rod element's mass
    velocity_collection: np.ndarray
        rod element's velocity
    external_forces: np.ndarray
        rod element's external forces

    Returns
    -------
    magnitude of the plane response
    """

    # Damping force response due to velocity towards the plane
    element_velocity = elastica.contact_utils._node_to_element_velocity(
        mass=mass, node_velocity_collection=velocity_collection)

    if len(face_idx_array) > 0:
        element_position_contacts = element_position[:, position_idx_array]
        contact_face_centers = faces_centers[:, face_idx_array]
        normals_on_elements = faces_normals[:, face_idx_array]
        radius_contacts = radius[position_idx_array]
        element_velocity_contacts = element_velocity[:, position_idx_array]

    else:
        element_position_contacts = element_position
        contact_face_centers = np.zeros_like(element_position)
        normals_on_elements = np.zeros_like(element_position)
        radius_contacts = radius
        element_velocity_contacts = element_velocity

    # Elastic force response due to penetration

    distance_from_plane = _batch_dot(
        normals_on_elements,
        (element_position_contacts - contact_face_centers))
    plane_penetration = (
        -np.abs(np.minimum(distance_from_plane - radius_contacts, 0.0))**1.5)
    elastic_force = -k * _batch_product_k_ik_to_ik(plane_penetration,
                                                   normals_on_elements)

    normal_component_of_element_velocity = _batch_dot(
        normals_on_elements, element_velocity_contacts)
    damping_force = -nu * _batch_product_k_ik_to_ik(
        normal_component_of_element_velocity, normals_on_elements)

    # Compute total plane response force
    plane_response_force_contacts = elastic_force + damping_force

    # Check if the rod elements are in contact with plane.
    no_contact_point_idx = np.where((distance_from_plane -
                                     radius_contacts) > surface_tol)[0]
    # If rod element does not have any contact with plane, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    plane_response_force_contacts[..., no_contact_point_idx] = 0.0

    plane_response_forces = np.zeros_like(external_forces)
    for i in range(len(position_idx_array)):
        plane_response_forces[:, position_idx_array[
            i]] += plane_response_force_contacts[:, i]

    # Update the external forces
    elastica.contact_utils._elements_to_nodes_inplace(plane_response_forces,
                                                      external_forces)
    return (
        _batch_norm(plane_response_force_contacts),
        no_contact_point_idx,
        normals_on_elements,
    )


class RodMeshSurfaceContactWithGridMethod(NoContact):
    """
    This class is for applying contact forces between rod-mesh_surface.
    First system is always rod and second system is always mesh_surface.

    Examples
    --------
    How to define contact between rod and mesh_surface.

    >>> simulator.detect_contact_between(rod, mesh_surface).using(
    ...    RodMeshSurfaceContactWithGridMethod,
    ...    k=1e4,
    ...    nu=10,
    ...    surface_tol=1e-2,
    ... )
    """

    def __init__(self,
                 k: float,
                 nu: float,
                 faces_grid: dict,
                 grid_size: float,
                 surface_tol=1e-4):
        """

        Parameters
        ----------
        k: float
            Stiffness coefficient between the plane and the rod-like object.
        nu: float
            Dissipation coefficient between the plane and the rod-like object.
        faces_grid: dict
            Dictionary containing the grid information of the mesh surface.
        grid_size: float
            Grid size of the mesh surface.
        surface_tol: float
            Penetration tolerance between the surface and the rod-like object.

        """
        super(RodMeshSurfaceContactWithGridMethod, self).__init__()
        # n_faces = faces.shape[-1]
        self.k = k
        self.nu = nu
        self.faces_grid = faces_grid
        self.grid_size = grid_size
        self.surface_tol = surface_tol

    def _check_systems_validity(
        self,
        system_one: SystemType,
        system_two: AllowedContactType,
        faces_grid: dict,
        grid_size: float,
    ) -> None:
        """
        This checks the contact order and type of a SystemType object and an AllowedContactType object.
        For the RodMeshSurfaceContact class first_system should be a rod and second_system should be a mesh_surface;
        morever, the imported grid's attributes should match imported rod-mesh_surface(in contact) grid's attributes.

        Parameters
        ----------
        system_one
            SystemType
        system_two
            AllowedContactType
        """
        if not issubclass(system_one.__class__, RodBase) or not issubclass(
                system_two.__class__, MeshSurface):
            raise TypeError(
                "Systems provided to the contact class have incorrect order/type. \n"
                " First system is {0} and second system is {1}. \n"
                " First system should be a rod, second should be a mesh surface"
                .format(system_one.__class__, system_two.__class__))

        elif not faces_grid["grid_size"] == grid_size:
            raise TypeError(
                "Imported grid size does not match with the current rod-mesh_surface grid size. "
            )

        elif not faces_grid["model_path"] == system_two.model_path:
            raise TypeError(
                "Imported grid's model path does not match with the current mesh_surface model path. "
            )

        elif not np.all(
                faces_grid["surface_reorient"] == system_two.mesh_orientation):
            raise TypeError(
                "Imported grid's surface orientation does not match with the current mesh_surface rientation. "
            )

    def apply_contact(self, system_one: RodType,
                      system_two: AllowedContactType) -> tuple:
        """
        In the case of contact with the plane, this function computes the plane reaction force on the element.

        Parameters
        ----------
        system_one: object
            Rod-like object.
        system_two: Surface
            Mesh surface.

        Returns
        -------
        plane_response_force_mag : numpy.ndarray
            1D (blocksize) array containing data with 'float' type.
            Magnitude of plane response force acting on rod-like object.
        no_contact_point_idx : numpy.ndarray
            1D (blocksize) array containing data with 'int' type.
            Index of rod-like object elements that are not in contact with the plane.
        """

        self.mesh_surface_faces = system_two.faces
        self.mesh_surface_x_min = np.min(self.mesh_surface_faces[0, :, :])
        self.mesh_surface_y_min = np.min(self.mesh_surface_faces[1, :, :])
        self.mesh_surface_face_normals = system_two.face_normals
        self.mesh_surface_face_centers = system_two.face_centers
        (
            self.position_idx_array,
            self.face_idx_array,
            self.element_position,
        ) = find_contact_faces_idx(
            self.faces_grid,
            self.mesh_surface_x_min,
            self.mesh_surface_y_min,
            self.grid_size,
            system_one.position_collection,
        )

        return  _calculate_contact_forces_rod_mesh_surface(
            self.mesh_surface_face_normals,
            self.mesh_surface_face_centers,
            self.element_position,
            self.position_idx_array,
            self.face_idx_array,
            self.surface_tol,
            self.k,
            self.nu,
            system_one.radius,
            system_one.mass,
            system_one.velocity_collection,
            system_one.external_forces,
        )


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
