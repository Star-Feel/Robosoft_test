__all__ = [
    "JoinableRodSphereContact",
    "RodMeshSurfaceContactWithGridMethod",
]
from typing import Union

import elastica
import numba
import numpy as np
from elastica import (
    AllowedContactType,
    NoContact,
    RodSphereContact,
    RodType,
    SystemType,
)
from elastica._linalg import _batch_dot, _batch_norm, _batch_product_k_ik_to_ik
from elastica.rod import RodBase

from .surface import MeshSurface


def find_contact_faces_idx_xy(
    faces_grid, x_min, y_min, grid_size, position_collection
):
    element_position = elastica.contact_utils._node_to_element_position(
        position_collection
    )
    n_element = element_position.shape[-1]

    # 预分配列表提升性能
    position_idx_list = []
    face_idx_list = []

    # 计算网格坐标（向量化计算）
    grid_coords = (
        element_position[0:2, :] - np.array([[x_min], [y_min]])
    ) / grid_size
    grid_x, grid_y = np.round(grid_coords[0]), np.round(grid_coords[1])

    # 定义安全查询函数
    def safe_get(x, y):
        return faces_grid.get((int(x), int(y)), np.array([], dtype=int))

    for i in range(n_element):
        # 查询四邻域网格
        queries = [
            safe_get(grid_x[i], grid_y[i]),  # 当前网格
            safe_get(grid_x[i] - 1, grid_y[i]),  # 左侧网格
            safe_get(grid_x[i] - 1, grid_y[i] - 1),  # 左下方网格
            safe_get(grid_x[i], grid_y[i] - 1)  # 下方网格
        ]

        # 合并并去重
        face_idx_element = np.unique(np.concatenate(queries))

        if face_idx_element.size > 0:
            # 记录有效接触对
            position_idx_list.extend([i] * len(face_idx_element))
            face_idx_list.extend(face_idx_element.tolist())

    # 转换为数组输出
    return (
        np.array(position_idx_list,
                 dtype=int), np.array(face_idx_list,
                                      dtype=int), element_position
    )


def find_contact_faces_idx_xyz(
    faces_grid, x_min, y_min, z_min, grid_size, position_collection
):
    # 转换节点坐标为元素中心坐标
    element_position = elastica.contact_utils._node_to_element_position(
        position_collection
    )
    n_element = element_position.shape[-1]

    # 预分配列表提升性能
    position_idx_list = []
    face_idx_list = []

    # 三维网格坐标计算（向量化）
    grid_coords = (
        element_position[:3, :] - np.array([[x_min], [y_min], [z_min]])
    ) / grid_size
    grid_x, grid_y, grid_z = np.round(grid_coords[0]
                                      ), np.round(grid_coords[1]
                                                  ), np.round(grid_coords[2])

    # 安全查询函数（三维版本）
    def safe_get_3d(x, y, z):
        return faces_grid.get((int(x), int(y), int(z)), np.array([],
                                                                 dtype=int))

    # 定义三维邻域偏移（3x3x3邻域）
    neighbor_offsets = [
        (0, 0, 0),  # 当前网格
        (-1, 0, 0),  # x-1
        (1, 0, 0),  # x+1
        (0, -1, 0),  # y-1
        (0, 1, 0),  # y+1
        (0, 0, -1),  # z-1
        (0, 0, 1),  # z+1
        # 可根据需要扩展更多邻域
    ]

    for i in range(n_element):
        queries = []
        # 遍历所有邻域偏移
        for dx, dy, dz in neighbor_offsets:
            x = grid_x[i] + dx
            y = grid_y[i] + dy
            z = grid_z[i] + dz
            queries.append(safe_get_3d(x, y, z))

        # 合并并去重
        face_idx_element = np.unique(np.concatenate(queries))

        if face_idx_element.size > 0:
            position_idx_list.extend([i] * len(face_idx_element))
            face_idx_list.extend(face_idx_element.tolist())

    return (
        np.array(position_idx_list,
                 dtype=int), np.array(face_idx_list,
                                      dtype=int), element_position
    )


@numba.njit(cache=True)
def surface_grid_numba_xy(
    faces, grid_size, face_x_left, face_x_right, face_y_down, face_y_up
):
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
                np.where(((face_y_down > y_up) + (face_y_up < y_down)
                          + (face_x_right < x_left)
                          + (face_x_left > x_right)) == 0)[0]
            ):
                faces_grid[(i, j)] = np.where(
                    ((face_y_down > y_up) + (face_y_up < y_down)
                     + (face_x_right < x_left) + (face_x_left > x_right)) == 0
                )[0]
    return faces_grid


@numba.njit(cache=True)
def surface_grid_numba_xyz(
    faces, grid_size, face_x_left, face_x_right, face_y_down, face_y_up,
    face_z_bottom, face_z_top
):
    """
    三维网格划分版本
    faces形状应为(3, m, n)，其中m为面数量，n为顶点数
    """
    # 计算三维空间范围
    x_min = np.min(faces[0, :, :])
    y_min = np.min(faces[1, :, :])
    z_min = np.min(faces[2, :, :])

    # 计算各方向网格数量
    n_x = int(np.ceil((np.max(faces[0, :, :]) - x_min) / grid_size))
    n_y = int(np.ceil((np.max(faces[1, :, :]) - y_min) / grid_size))
    n_z = int(np.ceil((np.max(faces[2, :, :]) - z_min) / grid_size))

    faces_grid = dict()

    # 三维网格遍历
    for i in range(n_x):
        x_l = x_min + i * grid_size
        x_r = x_min + (i + 1) * grid_size

        for j in range(n_y):
            y_d = y_min + j * grid_size
            y_u = y_min + (j + 1) * grid_size

            for k in range(n_z):
                z_b = z_min + k * grid_size
                z_t = z_min + (k + 1) * grid_size

                # 六方向非重叠判断
                mask = ((face_x_left > x_r) |  # 面在右侧
                        (face_x_right < x_l) |  # 面在左侧
                        (face_y_down > y_u) |  # 面在上方
                        (face_y_up < y_d) |  # 面在下方
                        (face_z_bottom > z_t) |  # 面在前方
                        (face_z_top < z_b)  # 面在后方
                        ) == 0  # 取反得到相交的面

                if np.any(mask):
                    faces_grid[(i, j, k)] = np.where(mask)[0]

    return faces_grid


def surface_grid_xy(faces, grid_size):
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
        surface_grid_numba_xy(
            faces, grid_size, face_x_left, face_x_right, face_y_down, face_y_up
        )
    )


def surface_grid_xyz(faces, grid_size):
    """
    Returns the faces_grid dictionary for rod-meshsurface contact
    This function only creates the faces_grid dict;
    the user must store the data in a binary file using pickle.dump
    """
    face_x_left = np.min(faces[0, :, :], axis=0)
    face_y_down = np.min(faces[1, :, :], axis=0)
    face_x_right = np.max(faces[0, :, :], axis=0)
    face_y_up = np.max(faces[1, :, :], axis=0)
    face_z_bottom = np.min(faces[2, :, :], axis=0)
    face_z_top = np.max(faces[2, :, :], axis=0)

    return dict(
        surface_grid_numba_xyz(
            faces,
            grid_size,
            face_x_left,
            face_x_right,
            face_y_down,
            face_y_up,
            face_z_bottom,
            face_z_top,
        )
    )


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
        mass=mass, node_velocity_collection=velocity_collection
    )

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
        (element_position_contacts - contact_face_centers)
    )
    plane_penetration = (
        -np.abs(np.minimum(distance_from_plane - radius_contacts, 0.0))**1.5
    )
    elastic_force = -k * _batch_product_k_ik_to_ik(
        plane_penetration, normals_on_elements
    )

    normal_component_of_element_velocity = _batch_dot(
        normals_on_elements, element_velocity_contacts
    )
    damping_force = -nu * _batch_product_k_ik_to_ik(
        normal_component_of_element_velocity, normals_on_elements
    )

    # Compute total plane response force
    plane_response_force_contacts = elastic_force + damping_force

    # Check if the rod elements are in contact with plane.
    no_contact_point_idx = np.where((distance_from_plane
                                     - radius_contacts) > surface_tol)[0]
    # If rod element does not have any contact with plane, plane cannot apply response
    # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
    plane_response_force_contacts[..., no_contact_point_idx] = 0.0

    plane_response_forces = np.zeros_like(external_forces)
    for i in range(len(position_idx_array)):
        plane_response_forces[:, position_idx_array[i]
                              ] += plane_response_force_contacts[:, i]

    # Update the external forces
    elastica.contact_utils._elements_to_nodes_inplace(
        plane_response_forces, external_forces
    )
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

    def __init__(
        self,
        k: float,
        nu: float,
        faces_grid: dict,
        grid_size: float,
        surface_tol=1e-4
    ):
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
            system_two.__class__, MeshSurface
        ):
            raise TypeError(
                "Systems provided to the contact class have incorrect order/type. \n"
                " First system is {0} and second system is {1}. \n"
                " First system should be a rod, second should be a mesh surface"
                .format(system_one.__class__, system_two.__class__)
            )

        elif not faces_grid["grid_size"] == grid_size:
            raise TypeError(
                "Imported grid size does not match with the current rod-mesh_surface grid size. "
            )

        elif not faces_grid["model_path"] == system_two.model_path:
            raise TypeError(
                "Imported grid's model path does not match with the current mesh_surface model path. "
            )

        elif not np.all(
            faces_grid["surface_reorient"] == system_two.mesh_orientation
        ):
            raise TypeError(
                "Imported grid's surface orientation does not match with the current mesh_surface rientation. "
            )

    def apply_contact(
        self, system_one: RodType, system_two: AllowedContactType
    ) -> tuple:
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
        self.mesh_surface_z_min = np.min(self.mesh_surface_faces[2, :, :])
        self.mesh_surface_face_normals = system_two.face_normals
        self.mesh_surface_face_centers = system_two.face_centers
        (
            self.position_idx_array,
            self.face_idx_array,
            self.element_position,
        ) = find_contact_faces_idx_xyz(
            self.faces_grid,
            self.mesh_surface_x_min,
            self.mesh_surface_y_min,
            self.mesh_surface_z_min,
            self.grid_size,
            system_one.position_collection,
        )

        return _calculate_contact_forces_rod_mesh_surface(
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


class RodMeshSurfaceContactWithGridMethodWithContactFlag(
    RodMeshSurfaceContactWithGridMethod
):

    def __init__(
        self,
        k: float,
        nu: float,
        faces_grid: dict,
        grid_size: float,
        surface_tol=1e-4,
        contact_flag: list[bool] = [False],
        contact_flag_id: int = 0,
    ):
        super().__init__(
            k,
            nu,
            faces_grid,
            grid_size,
            surface_tol=surface_tol,
        )
        self.contact_flag = contact_flag
        self.contact_flag_id = contact_flag_id

    def apply_contact(
        self, system_one: RodType, system_two: AllowedContactType
    ) -> tuple:
        pass


class JoinableRodSphereContact(RodSphereContact):

    def __init__(
        self,
        k: float,
        nu: float,
        velocity_damping_coefficient=0.0,
        friction_coefficient=0.0,
        index: Union[int, np.ndarray] = -1,
        action_flags: list[bool] = [False],
        attach_flags: list[bool] = None,
        flag_id: int = 0,
        collision: bool = True,
        eps: float = 1e-3
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
        self.action_flags = action_flags
        self.attach_flags = attach_flags
        self.flag_id = flag_id
        self.relative_distance = None
        self.relative_direction = None
        self.collision = collision
        self.eps = eps

    def _check_systems_validity(
        self,
        system_one,
        system_two,
    ) -> None:
        pass

    def _attach_check(
        self,
        system_one: RodType,
        system_two: AllowedContactType,
    ) -> bool:
        """
        Check if the rod is attached to the sphere.

        Parameters
        ----------
        system_one: object
            Rod object.
        system_two: object
            Sphere object.

        Returns
        -------
        bool
            True if the rod is attached to the sphere, False otherwise.
        """
        radias = system_two.radius
        center = system_two.position_collection
        rod_pos = system_one.position_collection
        if np.linalg.norm(rod_pos[..., -1]
                          - center[..., 0]) <= radias * (1 + self.eps):
            self.attach_flags[self.flag_id] = True
        else:
            self.attach_flags[self.flag_id] = False

    def apply_contact(
        self, system_one: RodType, system_two: AllowedContactType
    ) -> None:
        """
        Apply contact forces and torques between RodType object and Sphere object.

        Parameters
        ----------
        system_one: object
            Rod object.
        system_two: object
            Sphere object.

        """
        self._attach_check(system_one, system_two)
        if not self.action_flags[self.flag_id]:
            self.relative_distance = None
            self.relative_direction = None
            if self.collision:
                super().apply_contact(system_one, system_two)
        else:
            if self.relative_distance is None:

                self.relative_distance = np.linalg.norm(
                    system_two.position_collection[..., 0]
                    - system_one.position_collection[..., self.index]
                )

                system_one_direction = (
                    system_one.position_collection[..., self.index]
                    - system_one.position_collection[..., self.index - 1]
                )
                system_one_direction = system_one_direction / np.linalg.norm(
                    system_one_direction
                )
                system_two_direction = (
                    system_two.position_collection[..., 0]
                    - system_one.position_collection[..., self.index]
                )
                system_two_direction = system_two_direction / np.linalg.norm(
                    system_two_direction
                )

                self.relative_direction = (
                    system_two_direction - system_one_direction
                )

            else:
                system_one_direction = (
                    system_one.position_collection[..., self.index]
                    - system_one.position_collection[..., self.index - 1]
                )
                system_one_direction = system_one_direction / np.linalg.norm(
                    system_one_direction
                )
                system_two_direction = (
                    system_one_direction + self.relative_direction
                )
                system_two_direction = system_two_direction / np.linalg.norm(
                    system_two_direction
                )
                relative_position = (
                    system_two_direction * self.relative_distance
                )
                system_two.position_collection[..., 0] = (
                    system_one.position_collection[..., self.index]
                    + relative_position
                )

            system_two.velocity_collection[
                ..., 0] = system_one.velocity_collection[..., self.index]
            # print(system_two.velocity_collection[..., 0])
