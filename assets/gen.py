import os
import numpy as np
import trimesh
from stl import mesh


def create_cylinder(radius, height, segments, filename):
    cylinder = trimesh.creation.cylinder(radius, height, sections=segments)
    cylinder.export(filename)
    print(f"Cylinder STL saved to {filename}")


def create_cone(radius, height, segments, filename):
    cone = trimesh.creation.cone(radius, height, sections=segments)
    cone.export(filename)
    print(f"Cone STL saved to {filename}")


def create_sphere(radius, segments, filename):
    sphere = trimesh.creation.icosphere(subdivisions=segments, radius=radius)
    sphere.export(filename)
    print(f"Sphere STL saved to {filename}")


if __name__ == "__main__":
    # Parameters
    cylinder_radius = 1
    cylinder_height = 2
    cone_radius = 1
    cone_height = 2
    sphere_radius = 1
    segments = 32  # Number of segments for smoothness

    # Output directory
    output_dir = "/data/zyw/workshop/attempt/assets"
    os.makedirs(output_dir, exist_ok=True)

    # File names
    cylinder_file = os.path.join(output_dir, "cylinder.stl")
    cone_file = os.path.join(output_dir, "cone.stl")
    sphere_file = os.path.join(output_dir, "sphere.stl")

    # Generate STL files
    create_cylinder(cylinder_radius, cylinder_height, segments, cylinder_file)
    create_cone(cone_radius, cone_height, segments, cone_file)
    create_sphere(sphere_radius, segments, sphere_file)
