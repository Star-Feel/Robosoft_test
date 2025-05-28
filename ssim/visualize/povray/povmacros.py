""" POVray macros for pyelastica

This module includes utility methods to support POVray rendering.

"""
__all__ = [
    'pyelastica_rod',
    'render',
    'Stages',
]

import subprocess
from collections import UserDict, defaultdict
from .pov_objects import Camera, Light, Box, Sphere, StageObject, MeshObject


def pyelastica_rod(
    x,
    r,
    color="rgb<0.45,0.39,1>",
    transmit=0.0,
    interpolation="linear_spline",
    deform=None,
    tab="    ",
):
    """pyelastica_rod POVray script generator

    Generates povray sphere_sweep object in string.
    The rod is given with the element radius (r) and joint positions (x)

    Parameters
    ----------
    x : numpy array
        Position vector
        Expected shape: [num_time_step, 3, num_element]
    r : numpy array
        Radius vector
        Expected shape: [num_time_step, num_element]
    color : str
        Color of the rod (default: Purple <0.45,0.39,1>)
    transmit : float
        Transparency (0.0 to 1.0).
    interpolation : str
        Interpolation method for sphere_sweep
        Supporting type: 'linear_spline', 'b_spline', 'cubic_spline'
        (default: linear_spline)
    deform : str
        Additional object deformation
        Example: "scale<4,4,4> rotate<0,90,90> translate<2,0,4>"

    Returns
    -------
    cmd : string
        Povray script
    """

    assert interpolation in ["linear_spline", "b_spline", "cubic_spline"]
    tab = "    "

    # Parameters
    num_element = r.shape[0]

    lines = []
    lines.append("sphere_sweep {")
    lines.append(f"\t{interpolation} {num_element}")
    for i in range(num_element):
        lines.append(f"\t,<{x[0,i]},{x[1,i]},{x[2,i]}>,{r[i]}")
    lines.append("\ttexture{")
    lines.append("\t\tpigment{ color %s transmit %f }" % (color, transmit))
    lines.append("\t\tfinish{ phong 1 }")
    lines.append("\t}")
    if deform is not None:
        lines.append(f"\t{deform}")
    lines.append("\t}\n")

    cmd = "\n".join(lines)
    return cmd


def render(filename,
           width,
           height,
           antialias="on",
           quality=11,
           display="Off",
           pov_thread=4):

    # Define script path and image path
    script_file = filename + ".pov"
    image_file = filename + ".png"

    # Run Povray as subprocess
    cmds = [
        "povray",
        "+I" + script_file,
        "+O" + image_file,
        f"-H{height}",
        f"-W{width}",
        f"Work_Threads={pov_thread}",
        f"Antialias={antialias}",
        f"Quality={quality}",
        f"Display={display}",
    ]
    process = subprocess.Popen(cmds,
                               stderr=subprocess.PIPE,
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    _, stderr = process.communicate()

    # Check execution error
    if process.returncode:
        print(type(stderr), stderr)
        raise IOError("POVRay rendering failed with the following error: " +
                      stderr.decode("ascii"))


class Stages:

    def __init__(
        self,
        pre_scripts: str = "",
        post_scripts: str = "",
    ):
        self.pre_scripts = pre_scripts
        self.post_scripts = post_scripts
        self.cameras: UserDict[str, Camera] = {}
        self.lights: list[Light] = []
        self.stage_objects: UserDict[str, StageObject] = {}
        self._light_assign: UserDict[str, list[int]] = defaultdict(list)

    def add_camera(self, name, **kwargs):
        """Add camera (viewpoint)"""
        assert self.cameras.get(name) is None, "Camera name already exists"
        self.cameras[name] = Camera(**kwargs)

    def add_light(self, camera_name="All", **kwargs):
        """Add lighting and assign to camera
        Parameters
        ----------
        camera_id : int or list
            Assigned camera. [default=-1]
            If a list of camera_id is given, light is assigned for listed camera.
            If camera_id==-1, the lighting is assigned for all camera.
        """
        light_id = len(self.lights)
        self.lights.append(Light(**kwargs))
        if isinstance(camera_name, list) or isinstance(camera_name, tuple):
            camera_name = list(set(camera_name))
            for name in camera_name:
                self._light_assign[name].append(light_id)
        elif isinstance(camera_name, str):
            self._light_assign[camera_name].append(light_id)
        else:
            raise NotImplementedError("camera_name can only be a list or str")

    def add_stage_object(self, object_type, name, **kwargs):
        assert object_type in ["box", "sphere",
                               "mesh"], "Object type not supported"
        assert self.stage_objects.get(
            name) is None, "Object name already exists"
        if object_type == "box":
            self.stage_objects[name] = Box(**kwargs)
        elif object_type == "sphere":
            self.stage_objects[name] = Sphere(**kwargs)
        elif object_type == "mesh":
            self.stage_objects[name] = MeshObject(**kwargs)

    def generate_scripts(self):
        """Generate pov-ray script for all camera setup
        Returns
        -------
        scripts : list
            Return list of pov-scripts (string) that includes camera and assigned lightings.
        """
        scripts = {}
        for name, camera in self.cameras.items():
            light_ids = self._light_assign[name] + self._light_assign["All"]
            cmds = []
            cmds.append(self.pre_scripts)
            cmds.append(str(camera))  # Script camera
            for light_id in light_ids:  # Script Lightings
                cmds.append(str(self.lights[light_id]))
            for stage_object in self.stage_objects.values():
                cmds.append(str(stage_object))
            cmds.append(self.post_scripts)
            scripts[name] = "\n".join(cmds)
        return scripts

    def __len__(self):
        return len(self.cameras)
