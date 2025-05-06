import bpy
import re
import os
import math
import subprocess


class BlenderRenderer:
    def __init__(self, output_dir="renders"):
        self.set_environment()
        self.output_dir = output_dir

    def set_environment(self):
        # 清除当前场景
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # 获取或创建世界
        if bpy.data.worlds:
            world = bpy.data.worlds[0]
        else:
            world = bpy.data.worlds.new("World")
        
        # 确保场景有一个世界设置
        bpy.context.scene.world = world

        # 设置背景颜色 (使用节点系统，适用于 Blender 2.8+)
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = (0.1, 0.1, 0.1, 1.0)  # RGBA
        
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        env_light = bpy.context.object
        env_light.name = "EnvironmentLight"
        env_light.data.energy = 1.0  
        env_light.data.color = (0.5, 0.5, 0.5)  # 灰色光

    @staticmethod
    def coord_pov2blend(x, y, z):
        x_prime = -z 
        y_prime = y
        z_prime = x
        return x_prime, y_prime, z_prime

    def load_pov_settings(self, pov_file, obj_path=None):
        self.pov_file = pov_file
        with open(pov_file, 'r') as file:
            self.pov_content = file.read()
        
        self.set_camera()
        self.set_light()
        if obj_path:
            self.load_obj(obj_path)
        else:
            self.set_sphere()
        self.set_snake()

    def load_obj(self, obj_path):
        bpy.ops.wm.obj_import(filepath=obj_path)
        # 放置在球体对应位置，并设置材质
        sphere_matche = re.findall(r'sphere\s*{([^}]*)}', self.pov_content, re.DOTALL)[0]
        sphere_data_match = re.search(r'<([^>]*)>\s*,\s*([0-9.]+)', sphere_matche)
        if sphere_data_match:
            pos = map(float, sphere_data_match.group(1).split(','))
            pos = self.coord_pov2blend(*pos)
            obj = bpy.context.selected_objects[0]
            obj.location = tuple(pos)
            obj.rotation_euler = (math.radians(90), 0, math.radians(90))
            obj.scale = (1, 1, 1) 

    def set_camera(self):
        # 设置相机
        camera_match = re.search(r'camera\s*{([^}]*)}', self.pov_content, re.DOTALL)
        if camera_match:
            camera_text = camera_match.group(1)
            
            # 提取相机位置
            location_match = re.search(r'location\s*<([^>]*)>', camera_text)
            if location_match:
                x, y, z = map(float, location_match.group(1).split(','))
                x, y, z = self.coord_pov2blend(x, y, z)
                bpy.ops.object.camera_add(location=(x,y,z))
                # 获取刚创建的相机对象并设置为活动相机
                cam = bpy.context.object
                bpy.context.scene.camera = cam
            
            # 提取相机角度
            angle_match = re.search(r'angle\s*(\d+)', camera_text)
            if angle_match:
                angle = float(angle_match.group(1))
                # 转换 POV-Ray 角度为 Blender 焦距
                bpy.data.cameras['Camera'].lens = 16 / math.tan(math.radians(angle/2))
            
            # 提取相机朝向
            look_at_match = re.search(r'look_at\s*<([^>]*)>', camera_text)
            if look_at_match:
                x, y, z = map(float, look_at_match.group(1).split(','))
                x, y, z = self.coord_pov2blend(x, y, z)
                # 创建一个目标空物体
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=(x, y, z))
                target = bpy.context.object
                target.name = "CameraTarget"
                
                # 添加 Track To 约束
                cam = bpy.data.objects['Camera']
                constraint = cam.constraints.new('TRACK_TO')
                constraint.target = target
                constraint.track_axis = 'TRACK_NEGATIVE_Z'
                constraint.up_axis = 'UP_Y'

    def set_light(self):
        # 设置光源
        light_match = re.search(r'light_source\s*{([^}]*)}', self.pov_content, re.DOTALL)
        if light_match:
            light_text = light_match.group(1)
            
            # 提取光源位置
            position_match = re.search(r'<([^>]*)>', light_text)
            if position_match:
                x, y, z = map(float, position_match.group(1).split(','))
                x, y, z = self.coord_pov2blend(x, y, z)
                
                # 创建光源
                bpy.ops.object.light_add(type='POINT', location=(x, y, z))
                light = bpy.context.object
                light.name = "POVLight"

    def set_sphere(self):
        # 处理球体
        sphere_matches = re.findall(r'sphere\s*{([^}]*)}', self.pov_content, re.DOTALL)
        for i, sphere_text in enumerate(sphere_matches):
            # 提取球体位置和半径
            sphere_data_match = re.search(r'<([^>]*)>\s*,\s*([0-9.]+)', sphere_text)
            if sphere_data_match:
                pos = map(float, sphere_data_match.group(1).split(','))
                pos = self.coord_pov2blend(*pos)
                radius = float(sphere_data_match.group(2))
                
                # 创建球体
                bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=tuple(pos))
                sphere = bpy.context.object
                sphere.name = f"Sphere_{i}"
                
                # 提取材质颜色
                color_match = re.search(r'color\s*[a-zA-Z]+|color\s*rgb\s*<([^>]*)>', sphere_text)
                if color_match and color_match.group(1):
                    r, g, b = map(float, color_match.group(1).split(','))
                    
                    # 创建材质
                    material = bpy.data.materials.new(name=f"SphereMaterial_{i}")
                    material.use_nodes = True
                    bsdf = material.node_tree.nodes["Principled BSDF"]
                    bsdf.inputs["Base Color"].default_value = (r, g, b, 1)
                    
                    # 指定材质
                    if sphere.data.materials:
                        sphere.data.materials[0] = material
                    else:
                        sphere.data.materials.append(material)

    def set_snake(self):
        # 处理球扫描（sphere_sweep）
        sphere_sweep_matches = re.findall(r'sphere_sweep\s*{([^}]*)}', self.pov_content, re.DOTALL)
        for i, sweep_text in enumerate(sphere_sweep_matches):
            # 提取类型和点数
            header_match = re.search(r'(linear_spline|cubic_spline|b_spline)\s*(\d+)', sweep_text)
            if header_match:
                spline_type = header_match.group(1)
                num_points = int(header_match.group(2))
                
                # 提取所有点和半径
                points = []
                point_matches = re.findall(r',<([^>]*)>\s*,\s*([0-9.e+-]+)', sweep_text)
                for point_match in point_matches:
                    pos = tuple(map(float, point_match[0].split(',')))
                    radius = float(point_match[1])
                    points.append((pos, radius))
                
                # 创建曲线
                curve_data = bpy.data.curves.new('SweepCurve', 'CURVE')
                curve_data.dimensions = '3D'
                curve_data.resolution_u = 12
                curve_data.bevel_depth = 0.01  # 先设置一个小值，后面会根据每个点调整
                
                # 创建样条
                polyline = curve_data.splines.new('BEZIER')
                polyline.bezier_points.add(len(points)-1)
                
                # 设置点位置
                for idx, (pos, radius) in enumerate(points):
                    pos = self.coord_pov2blend(*pos)
                    polyline.bezier_points[idx].co = pos
                    polyline.bezier_points[idx].handle_left_type = 'AUTO'
                    polyline.bezier_points[idx].handle_right_type = 'AUTO'
                
                # 创建曲线对象
                curve_obj = bpy.data.objects.new('SphereSweep_' + str(i), curve_data)
                bpy.context.collection.objects.link(curve_obj)
                
                # 提取材质颜色
                color_match = re.search(r'pigment\s*{\s*color\s*rgb\s*<([^>]*)>', sweep_text)
                if color_match:
                    r, g, b = map(float, color_match.group(1).split(','))
                    
                    # 创建材质
                    material = bpy.data.materials.new(name=f"SweepMaterial_{i}")
                    material.use_nodes = True
                    bsdf = material.node_tree.nodes["Principled BSDF"]
                    bsdf.inputs["Base Color"].default_value = (r, g, b, 1)
                    
                    # 指定材质
                    curve_obj.data.materials.append(material)

    def render(self):
        # 保存 Blend 文件
        pov_index = os.path.splitext(self.pov_file)[0].split('/')[-1]
        output_blend =  os.path.splitext(self.pov_file)[0] + ".blend"
        bpy.ops.wm.save_as_mainfile(filepath=output_blend)
        print(f"转换完成。已保存为：{output_blend}")

        # 渲染图像
        blender_cli = f"blender -b {output_blend} -o renders/{pov_index} -f 1"
        subprocess.run(blender_cli, shell=True)


if __name__ == "__main__":
    # POV 文件路径
    import time
    start_time = time.time()
    pov_file = "/home/wrp/SoftRoboticaSimulator/work_dirs/povray_continuum_snake/diag/frame_00300.pov"
    obj_path = "/home/wrp/SoftRoboticaSimulator/assets/lamed_chair/Lamed_chair.obj"

    pov_settings = BlenderRenderer()
    pov_settings.load_pov_settings(pov_file, obj_path)
    pov_settings.render()
    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f}秒")