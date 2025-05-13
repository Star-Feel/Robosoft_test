import numpy as np
from .ssim.components import MeshSurface

mesh_path = '/data/zyw/workshop/PyElastica/tests/cube.stl'
center = np.array([0.0, 0.0, 3.0])
scale = np.array([1., 1., 1.])
rotate = np.array([0., 0., 0.])

mesh = MeshSurface(mesh_path)
mesh.translate(center)
mesh.scale(scale)
mesh.rotate(np.array([1, 0, 0]), rotate[0])
mesh.rotate(np.array([0, 1, 0]), rotate[1])
mesh.rotate(np.array([0, 0, 1]), rotate[2])

objects_plot = []
# 将网格表面绘制为三角形面片
vertices = obj.face_centers
faces = obj.faces
for face in faces:
    # 获取面片的顶点坐标
    face_vertices = vertices[face]
    # 创建三角形面片
    triangle = ax.plot_trisurf(
        face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2],
        color='g', alpha=0.6
    )
    objects_plot.append(triangle)



