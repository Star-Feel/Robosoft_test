from stl import mesh


def demo_mesh(mesh_path):
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(mesh_path)

    # Print the number of vectors (triangles) in the mesh
    print(f"Number of triangles in the mesh: {len(stl_mesh.vectors)}")

    # Print the first triangle's vertices
    print("First triangle vertices:")
    print(stl_mesh.vectors[0])

    # Print the bounding box of the mesh
    x_min, x_max = stl_mesh.vectors[:, :, 0].min(), stl_mesh.vectors[:, :, 0].max()
    y_min, y_max = stl_mesh.vectors[:, :, 1].min(), stl_mesh.vectors[:, :, 1].max()
    z_min, z_max = stl_mesh.vectors[:, :, 2].min(), stl_mesh.vectors[:, :, 2].max()

    print(
        f"Bounding box: ({x_min}, {y_min}, {z_min}) to ({x_max}, {y_max}, {z_max})"
    )


mesh_path = '/data/zyw/workshop/attempt/assets/cube.stl'
demo_mesh(mesh_path)

mesh_path = '/data/zyw/workshop/attempt/assets/cylinder.stl'
demo_mesh(mesh_path)
mesh_path = '/data/zyw/workshop/attempt/assets/cone.stl'
demo_mesh(mesh_path)
mesh_path = '/data/zyw/workshop/attempt/assets/sphere.stl'
demo_mesh(mesh_path)
