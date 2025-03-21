import elastica as ea
import numpy as np


def is_contact(sphere: ea.Sphere, rod: ea.CosseratRod):
    radias = sphere.radius
    center = sphere.position_collection
    rod_pos = rod.position_collection
    flag = False
    if np.linalg.norm(rod_pos[..., -1] -
                      center[..., 0]) <= radias * (1 + 1e-3):
        flag = True
    return flag
