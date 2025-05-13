import elastica as ea
import numpy as np
import yaml
import json


def save_json(data, file_path: str, **kwargs):
    with open(file_path, "w") as f:
        json.dump(data, f, **kwargs)

def load_yaml(file_path: str):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data

def save_yaml(data, file_path: str):
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

def is_contact(sphere: ea.Sphere, rod: ea.CosseratRod):
    radias = sphere.radius
    center = sphere.position_collection
    rod_pos = rod.position_collection
    flag = False
    if np.linalg.norm(rod_pos[..., -1] -
                      center[..., 0]) <= radias * (1 + 1e-3):
        flag = True
    return flag


def convert_lists_to_arrays(d):
    for k, v in d.items():
        if isinstance(v, list):
            if all(isinstance(i, (int, float)) for i in v):
                d[k] = np.array(v)
            else:
                for item in v:
                    if isinstance(item, dict):
                        convert_lists_to_arrays(item)
        elif isinstance(v, dict):
            convert_lists_to_arrays(v)


def compute_rotation_matrix(theta: np.ndarray) -> np.ndarray:

    Rx = np.array([[1, 0, 0], [0, np.cos(theta[0]), -np.sin(theta[0])],
                   [0, np.sin(theta[0]), np.cos(theta[0])]])

    Ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])], [0, 1, 0],
                   [-np.sin(theta[1]), 0,
                    np.cos(theta[1])]])

    Rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                   [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx

def compute_quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    assert matrix.shape == (3, 3), "输入必须是 3x3 的旋转矩阵"

    # 提取矩阵的元素
    m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    # 计算四元数
    trace = m00 + m11 + m22  # 旋转矩阵的迹
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


# def compute_rotation_matrix(theta):
#     R = np.array([
#         [
#             -np.sin(theta[1]),
#             np.sin(theta[0]) * np.cos(theta[1]),
#             np.cos(theta[0]) * np.cos(theta[1]),
#         ],
#         [
#             np.cos(theta[1]) * np.cos(theta[2]),
#             np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2]) -
#             np.sin(theta[2]) * np.cos(theta[0]),
#             np.sin(theta[1]) * np.cos(theta[0]) * np.cos(theta[2]) +
#             np.sin(theta[0]) * np.sin(theta[2]),
#         ],
#         [
#             np.sin(theta[2]) * np.cos(theta[1]),
#             np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2]) +
#             np.cos(theta[0]) * np.cos(theta[2]),
#             np.sin(theta[1]) * np.sin(theta[2]) * np.cos(theta[0]) -
#             np.sin(theta[0]) * np.cos(theta[2]),
#         ],
#     ])
#     return R


# def compute_quaternion_from_matrix(Q):
#     # Compute target tip orientation using quaternions.
#     # We add target and arm tip orientations difference to reward function.
#     qw = np.sqrt(1 + Q[0, 0] + Q[1, 1] + Q[2, 2]) / 2
#     qx = (Q[2, 1] - Q[1, 2]) / (4 * qw)
#     qy = (Q[0, 2] - Q[2, 0]) / (4 * qw)
#     qz = (Q[1, 0] - Q[0, 1]) / (4 * qw)
#     return np.array([qw, qx, qy, qz])


def isnan_check(array: np.ndarray) -> bool:
    """
    This function checks if there is any nan inside the array.
    If there is nan, it returns True boolean
    Parameters
    ----------
    array

    Returns
    -------
    Notes
    -----
    Micro benchmark results showed that for a block size of 100, using timeit
    Numba version: 479 ns ± 6.49 ns per loop
    This version: 2.24 µs ± 96.1 ns per loop
    """
    return np.isnan(array).any()


if __name__ == "__main__":
    theta = np.array([0, 0, 0])
    w(theta)
