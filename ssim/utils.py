import elastica as ea
import numpy as np
import yaml
import json


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


def load_yaml(file_path: str):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(data, file_path: str):
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
