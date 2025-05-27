class StageObject:
    """Template for stage objects

    Objects (camera and light) is defined as an object in order to
    manipulate (translate or rotate) them during the rendering.

    Attributes
    ----------
    str : str
        String representation of object.
        The placeholder exist to avoid rescripting.

    Methods
    -------
    _color2str : str
        Change triplet tuple (or list) of color into rgb string.
    _position2str : str
        Change triplet tuple (or list) of position vector into string.
    """

    def __init__(self):
        self.str = ""
        self.update_script()

    def update_script(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        pass

    def __str__(self):
        return self.str

    def _color2str(self, color):
        if isinstance(color, str):
            return color
        elif isinstance(color, list) and len(color) == 3:
            # RGB
            return "rgb<{},{},{}>".format(*color)
        else:
            raise NotImplementedError(
                "Only string-type color or RGB input is implemented")

    def _position2str(self, position):
        assert len(position) == 3
        return "<{},{},{}>".format(*position)

    def _matrix2str(self, matrix):
        assert len(matrix) == 9
        return "<{},{},{},{},{},{},{},{},{},0,0,0>".format(*matrix)


class Camera(StageObject):
    """Camera object

    http://www.povray.org/documentation/view/3.7.0/246/

    Attributes
    ----------
    location : list or tuple
        Position vector of camera location. (length=3)
    angle : int
        Camera angle
    look_at : list or tuple
        Position vector of the location where camera points to (length=3)
    name : str
        Name of the view-point.
    sky : list or tuple
        Tilt of the camera (length=3) [default=[0,1,0]]
    """

    def __init__(self, location, angle, look_at, sky=(0, 1, 0)):
        self.location = location
        self.angle = angle
        self.look_at = look_at
        self.sky = sky
        super().__init__()

    def update(
        self,
        location=None,
        angle=None,
        look_at=None,
        sky=None,
    ):
        if location is not None:
            self.location = location
        if angle is not None:
            self.angle = angle
        if look_at is not None:
            self.look_at = look_at
        if sky is not None:
            self.sky = sky
        self.update_script()

    def update_script(self):
        location = self._position2str(self.location)
        look_at = self._position2str(self.look_at)
        sky = self._position2str(self.sky)
        cmds = []
        cmds.append("camera{")
        cmds.append(f"\tlocation {location}")
        cmds.append(f"\tangle {self.angle}")
        cmds.append(f"\tlook_at {look_at}")
        cmds.append(f"\tsky {sky}")
        cmds.append("\tright x*image_width/image_height")
        cmds.append("}")
        self.str = "\n".join(cmds)


class Light(StageObject):
    """Light object

    Attributes
    ----------
    position : list or tuple
        Position vector of light location. (length=3)
    color : str or list
        Color of the light.
        Both string form of color or rgb (normalized) form is supported.
        Example) color='White', color=[1,1,1]
    """

    def __init__(self, position, color):
        self.position = position
        self.color = color
        super().__init__()

    def update_script(self):
        position = self._position2str(self.position)
        color = self._color2str(self.color)
        cmds = []
        cmds.append("light_source{")
        cmds.append(f"\t{position}")
        cmds.append(f"\tcolor {color}")
        cmds.append("}")
        self.str = "\n".join(cmds)


class Box(StageObject):

    def __init__(self,
                 name,
                 min_corner,
                 max_corner,
                 rotate=[0, 0, 0],
                 texture='T_Stone25',
                 scale=4):
        self.name = name
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.rotate = rotate
        self.texture = texture
        self.scale = scale
        super().__init__()

    def update_script(self):
        min_corner = self._position2str(self.min_corner)
        max_corner = self._position2str(self.max_corner)
        rotate = self._position2str(self.rotate)
        cmds = []
        cmds.append("box {")
        cmds.append(f"\t{min_corner}")
        cmds.append(f"\t{max_corner}")
        cmds.append("\ttexture {")
        cmds.append(f"\t\t{self.texture}")
        cmds.append(f"\t\tscale {self.scale}")
        cmds.append("\t}")
        cmds.append(f"\trotate {rotate}")
        cmds.append("}")
        self.str = "\n".join(cmds)


class Sphere(StageObject):

    def __init__(self, position, radius, shape, color='Yellow'):
        self.position = position
        self.radius = radius
        self.color = color
        self.shape = shape
        super().__init__()

    def update_script(self):
        position = self._position2str(self.position)
        cmds = []
        cmds.append("sphere {")
        cmds.append(f"\t{position}, {self.radius}")
        cmds.append("\ttexture {")
        cmds.append("\t\tpigment {")
        cmds.append(f"\t\t\tcolor {self.color}")
        cmds.append(f"\t\t\tshape {self.shape}")
        cmds.append("\t\t}")
        cmds.append("\t}")
        cmds.append("}")
        self.str = "\n".join(cmds)


class MeshObject(StageObject):

    def __init__(
        self,
        mesh_name,
        position,
        scale,
        shape,
        matrix=[1, 0, 0, 0, 1, 0, 0, 0, 1],
    ):
        self.mesh_name = mesh_name
        self.position = position
        self.matrix = matrix
        self.scale = scale
        self.shape = shape
        super().__init__()

    def update_script(self):
        position = self._position2str(self.position)
        matrix = self._matrix2str(self.matrix)
        cmds = []
        cmds.append("object {")
        cmds.append(f"\t{self.mesh_name}")
        cmds.append(f"\tscale {self.scale}")
        cmds.append(f"\tmatrix {matrix}")
        cmds.append(f"\ttranslate {position}")
        cmds.append(f"\tshape {self.shape}")
        cmds.append("}")
        self.str = "\n".join(cmds)
