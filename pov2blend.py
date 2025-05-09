import bpy
import re
import os
import math
import time
import tqdm
import mathutils

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
            bg_node.inputs['Color'].default_value = (0.2, 0.2, 0.2, 1.0)  # RGBA

        # 创建顶部环境灯光
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
        top_light = bpy.context.object
        top_light.name = "TopAmbientLight"
        top_light.data.energy = 300
        top_light.data.size = 15  # 大面积光源
        top_light.data.color = (1.0, 1.0, 1.0)  # 纯白色
        
        # scene settings
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE_NEXT'                     # 使用 Eevee 引擎
        scene.eevee.use_gtao = True                               # 开启AO。（默认AO是关闭的）
        scene.render.image_settings.file_format = 'PNG'           # 保存格式 png
        scene.render.image_settings.color_mode = 'RGB'            # 保存 RGB 通道，不要alpha通道
        scene.render.image_settings.color_depth = '8'             # 8位颜色
        scene.display_settings.display_device = 'sRGB'            # 窗口显示时，使用sRGB颜色格式编码
        scene.view_settings.view_transform = 'Filmic'             # 窗口显示时，使用Filmic颜色变换
        scene.sequencer_colorspace_settings.name = 'Filmic sRGB'  # 保存图片时使用 Filmmic + sRGB
        scene.render.film_transparent = False                     # 背景不透明（纯色或者环境光贴图）
        scene.render.resolution_percentage = 25
        scene.eevee.taa_render_samples = 8
        scene.eevee.taa_samples = 0
        scene.eevee.gi_diffuse_bounces = 0

    @staticmethod
    def coord_pov2blend(x, y, z):
        x_prime = z 
        y_prime = x
        z_prime = y
        return x_prime, y_prime, z_prime

    def batch_rendering(self, pov_dir, output_dir):
        """批量渲染指定目录中的所有 POV 文件。
        
        Args:
            pov_dir (str): 包含 POV 文件的目录路径
            output_dir (str): 渲染输出的目录路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有 POV 文件并按帧号排序
        pov_files = [f for f in os.listdir(pov_dir) if f.endswith('.pov')]
        # 从文件名中提取帧号并排序
        frame_pattern = re.compile(r'frame_(\d+)\.pov')
        pov_files.sort(key=lambda f: int(frame_pattern.search(f).group(1)) if frame_pattern.search(f) else 0)
        
        if not pov_files:
            print(f"警告: 在 {pov_dir} 中未找到 POV 文件")
            return
        
        print(f"找到 {len(pov_files)} 个 POV 文件进行批量渲染")
        
        # 加载第一个文件来设置相机、光源和静态对象
        first_pov = os.path.join(pov_dir, pov_files[0])
        self.load_pov_settings(first_pov)
        
        # 处理每个 POV 文件
        total_start_time = time.time()
        for i, pov_file in tqdm.tqdm(enumerate(pov_files)):
            pov_path = os.path.join(pov_dir, pov_file)
            frame_match = frame_pattern.search(pov_file)
            frame_num = int(frame_match.group(1)) if frame_match else i
            
            print(f"rendering {pov_file} ({i+1}/{len(pov_files)})")
            
            # 只更新蛇的部分，保留相机和其他静态物体
            self.update_snake_only(pov_path)
            
            # 渲染当前帧
            bpy.context.scene.frame_set(frame_num)
            bpy.ops.render.render()
            
            # 保存渲染结果
            output_file = os.path.join(output_dir, f"frame_{frame_num:05d}.png")
            bpy.data.images["Render Result"].save_render(output_file)
            
        
        total_time = time.time() - total_start_time
        print(f"批量渲染完成，总用时: {total_time:.2f}秒，平均每帧: {total_time/len(pov_files):.2f}秒")

    def update_snake_only(self, pov_file):
        """只更新蛇的部分，保留其他场景元素不变。
        
        Args:
            pov_file (str): POV 文件路径
        """
        # 读取新的 POV 文件内容
        with open(pov_file, 'r') as file:
            self.pov_content = file.read()
        
        # 删除所有现有的蛇对象
        for obj in bpy.data.objects:
            if obj.name.startswith('SphereSweep_'):
                bpy.data.objects.remove(obj, do_unlink=True)
        
        # 重新创建蛇
        self.set_snake()

    def load_pov_settings(self, pov_file, obj_path="/data/wjs/wrp/SoftRoboticaSimulator/assets/basketball/BasketBall.obj"):
        self.pov_file = pov_file
        with open(pov_file, 'r') as file:
            self.pov_content = file.read()
        
        self.set_camera()
        # self.set_light()
        if obj_path:
            self.load_obj(obj_path)
        else:
            self.set_sphere()
        self.set_snake()

    def load_obj(self, obj_path):
        bpy.ops.wm.obj_import(filepath=obj_path)
        # 放置在球体对应位置，并设置材质
        sphere_match = re.findall(r'sphere\s*{([^}]*)}', self.pov_content, re.DOTALL)[0]
        sphere_data_match = re.search(r'<([^>]*)>\s*,\s*([0-9.]+)', sphere_match)
        if sphere_data_match:
            pos = map(float, sphere_data_match.group(1).split(','))
            pos = self.coord_pov2blend(*pos)
            radius = float(sphere_data_match.group(2))

            obj = bpy.context.selected_objects[0]
            obj.location = tuple(pos)
            obj.rotation_euler = (math.radians(90), 0, math.radians(90))
            # 如何将物体限制在球体内？

            # 计算物体边界盒的对角线长度（作为"直径"）
            local_bbox_corners = [mathutils.Vector(corner) for corner in obj.bound_box]
            max_dimension = max(
                (local_bbox_corners[0] - local_bbox_corners[6]).length,
                (local_bbox_corners[1] - local_bbox_corners[7]).length,
                (local_bbox_corners[2] - local_bbox_corners[4]).length
            )
            
            # 计算需要的缩放系数来使物体恰好适合球体
            # 使用直径的一半（半径）与球体半径比较
            scale_factor = (radius * 2) / max_dimension
            
            # 应用缩放（略微减小以确保不会超出）
            # 0.95是安全系数，可以根据需要调整
            safe_scale = scale_factor * 0.95
            obj.scale = (safe_scale, safe_scale, safe_scale)
            
            # 将物体位置设回球体中心
            obj.location = tuple(pos) 

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
                light.data.energy = 1000  # 设置光源强度

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
                
                # 提取所有点和半径
                points = []
                point_matches = re.findall(r',<([^>]*)>\s*,\s*([0-9.e+-]+)', sweep_text)
                for point_match in point_matches:
                    pos = tuple(map(float, point_match[0].split(',')))
                    pos = self.coord_pov2blend(*pos)
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
        bpy.context.scene.camera = bpy.data.objects['Camera']
        bpy.context.scene.frame_set(0)

        bpy.ops.render.render()
        bpy.data.images["Render Result"].save_render(f"renders/frame_000000.png")


if __name__ == "__main__":
    # POV 文件路径
    import time
    pov_dir = "/data/wjs/wrp/SoftRoboticaSimulator/work_dirs/povray_continuum_snake/diag"
    pov_file = "/data/wjs/wrp/SoftRoboticaSimulator/work_dirs/povray_continuum_snake/diag/frame_00000.pov"
    # obj_path = "/data/wjs/wrp/SoftRoboticaSimulator/assets/lamed_chair/Lamed_chair.obj"
    obj_path = "/data/wjs/wrp/SoftRoboticaSimulator/assets/basketball/BasketBall.obj"
    out_dir = "/data/wjs/wrp/SoftRoboticaSimulator/renders"

    pov_settings = BlenderRenderer()
    start_time = time.time()
    # pov_settings.load_pov_settings(pov_file, obj_path)\
    pov_settings.batch_rendering(pov_dir, out_dir)
    
    s_time = time.time()
    print("render time: ", time.time() - s_time)
    
    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f}秒")