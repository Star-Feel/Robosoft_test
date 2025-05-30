import math
import os
import re
import time

import bpy
import mathutils
import tqdm
import numpy as np

ASSET_PATHS = {
    "Obj": "./scene_assets/living_room/soccer/Obj.obj",
    "Tea_Cup": "./scene_assets/living_room/Tea_Cup/Tea_Cup.obj",
    "BasketBall": "./scene_assets/living_room/basketball/BasketBall.obj",
    "Coffee_cup_withe_": "./scene_assets/living_room/Coffee_cup_withe_obj/Coffee_cup_withe_.obj",
    "pillows_obj": "./scene_assets/living_room/OBJ_PILLOWS/pillows_obj.obj",
    "Book_by_Peter_Iliev_obj": "./scene_assets/living_room/Book_by_Peter_Iliev_obj/Book_by_Peter_Iliev_obj.obj",
    "Cone_Buoy": "./scene_assets/living_room/Cone_Buoy/conbyfr.obj",
    "Cone_Buoy_2": "./scene_assets/living_room/Cone_Buoy_2/conbyfr2.obj",
}


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
            bg_node.inputs['Color'].default_value = (
                1.0, 1.0, 1.0, 1.0
            )  # RGBA

        # 创建顶部环境灯光
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        top_light = bpy.context.object
        top_light.name = "SunLight"
        top_light.data.energy = 5
        top_light.data.color = (1.0, 1.0, 1.0)
        top_light.data.shadow_soft_size = 0.5  # 软阴影大小
        top_light.rotation_euler = (0.5, 0.0, 0.0)

        # bpy.ops.object.light_add(type='AREA', location=(0, 0, 10))
        # top_light = bpy.context.object
        # top_light.name = "TopAmbientLight"
        # top_light.data.energy = 300
        # top_light.data.size = 15  # 大面积光源
        # top_light.data.color = (1.0, 1.0, 1.0)  # 纯白色
        # top_light.data.use_shadow = True

        # bpy.ops.object.light_add(type='SPOT', location=(0, -2, 5), rotation=(math.radians(15), 0, 0))
        # spot_light = bpy.context.object
        # spot_light.name = "SpotLight"
        # spot_light.data.energy = 1000  # 增加能量
        # spot_light.data.spot_size = math.radians(60)  # 聚光灯的角度
        # spot_light.data.spot_blend = 0.15  # 聚光灯的边缘柔和度

        # plane settings
        bpy.ops.mesh.primitive_plane_add(
            size=20, enter_editmode=False, align='WORLD', location=(0, -5, 0)
        )
        plane = bpy.context.object
        plane.name = "GroundPlane"
        material = bpy.data.materials.new(name="GroundMaterial")
        material.diffuse_color = (0.5, 0.5, 0, 1.0)  # 设置为白色
        plane.data.materials.append(material)
        plane.rotation_euler[1] = math.radians(89)  # 将角度转换为弧度

        
        # scene settings
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE_NEXT'  # 使用 Eevee 引擎
        scene.eevee.use_gtao = True  # 开启AO。（默认AO是关闭的）
        scene.render.image_settings.file_format = 'PNG'  # 保存格式 png
        scene.render.image_settings.color_mode = 'RGB'  # 保存 RGB 通道，不要alpha通道
        scene.render.image_settings.color_depth = '8'  # 8位颜色
        scene.display_settings.display_device = 'sRGB'  # 窗口显示时，使用sRGB颜色格式编码
        scene.view_settings.view_transform = 'Filmic'  # 窗口显示时，使用Filmic颜色变换
        scene.sequencer_colorspace_settings.name = 'Filmic sRGB'  # 保存图片时使用 Filmmic + sRGB
        scene.render.film_transparent = False  # 背景不透明（纯色或者环境光贴图）
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1280
        # scene.render.resolution_percentage = 40
        scene.eevee.taa_render_samples = 2
        scene.eevee.taa_samples = 0
        scene.eevee.gi_diffuse_bounces = 0

    @staticmethod
    def coord_pov2blend(x, y, z):
        x_prime = z
        y_prime = x
        z_prime = y
        return x_prime, y_prime, z_prime

    def batch_rendering(self, pov_dir, output_dir, target_id):
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
        pov_files.sort(
            key=lambda f: int(frame_pattern.search(f).group(1))
            if frame_pattern.search(f) else 0
        )

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

            # 只更新软体机器人和目标物体的部分，保留相机和其他静态物体
            
            self.update_snake_only(pov_path)

            # 渲染当前帧
            bpy.context.scene.frame_set(frame_num)
            bpy.ops.render.render()

            # 保存渲染结果
            output_file = os.path.join(
                output_dir, f"frame_{frame_num:05d}.png"
            )
            bpy.data.images["Render Result"].save_render(output_file)

        total_time = time.time() - total_start_time
        print(
            f"批量渲染完成，总用时: {total_time:.2f}秒，平均每帧: {total_time/len(pov_files):.2f}秒"
        )

    def single_step_rendering(
        self,
        current_step: int,
        pov_script: dict[str, str],
        output_dir: str,
        save_img: bool = True,
    ) -> np.ndarray:

        os.makedirs(output_dir, exist_ok=True)

        # 加载文件来设置相机、光源和静态对象
        self.load_pov_settings(None, pov_script)

        # 只更新蛇的部分，保留相机和其他静态物体
        self.update_snake_only(None, pov_script)

        # 渲染当前帧
        bpy.context.scene.frame_set(current_step)
        bpy.ops.render.render(write_still=False)
        render_result = bpy.data.images["Render Result"]

        output_file = os.path.join(output_dir, "temp.png")
        render_result.save_render(output_file)
        # # 读取保存的渲染结果为numpy数组
        # # 使用 PIL 打开保存的渲染结果
        # with Image.open(output_file) as img:
        #     img_array = np.array(img)
        # with open(output_file, "rb") as f:
        #     img_data = np.frombuffer(f.read(), dtype=np.uint8)
        # img_array = np.array(img_data).reshape(height, width, 4)
        # return img_array

    def update_snake_only(self, pov_file):
        """只更新蛇的部分，保留其他场景元素不变。

        Args:
            pov_file (str): POV 文件路径
        """
        # 读取新的 POV 文件内容
        with open(pov_file, 'r') as file:
            self.pov_content = file.read()

        # 删除所有现有的杆对象
        for idx, obj in enumerate(bpy.data.objects):
            if obj.name.startswith('SphereSweep_'):
                bpy.data.objects.remove(obj, do_unlink=True)

        # 重新创建杆
        self.set_snake()

    def load_pov_settings(self, pov_file, obj_path=None):
        self.pov_file = pov_file
        with open(pov_file, 'r') as file:
            self.pov_content = file.read()

        self.set_camera()
        # self.set_light()
        self.load_obj(obj_path)
        self.set_snake()

    def load_obj(self, obj_path=None):
        # obj_path == None, then load obj randomly
        sphere_matchs = re.findall(
            r'sphere\s*{([^}]*)}', self.pov_content, re.DOTALL
        )
        mesh_matchs = re.findall(
            r'object\s*\{\s*cube_mesh[^}]*\}', self.pov_content, re.DOTALL
        )

        if obj_path is None:

            for idx, sphere_match in enumerate(sphere_matchs):

                shape_match = re.search(r'shape\s+([^\s>]+)', sphere_match)
                shape = shape_match.group(1)
                obj_path = ASSET_PATHS[shape]

                bpy.ops.wm.obj_import(filepath=obj_path)
                for idx, obj in enumerate(bpy.context.selected_objects):
                    if idx == 0:
                        continue
                    else:
                        bpy.data.objects.remove(obj, do_unlink=True)
            for idx, mesh_match in enumerate(mesh_matchs):
                # obj_chosen = random.choice(obj_list)

                mesh_shape = re.search(r'shape\s+([^\s>]+)', mesh_match)
                shape = mesh_shape.group(1)
                if shape == "Cone_Buoy":
                    tar_idx = 1
                elif shape == "Cone_Buoy_2":
                    tar_idx = 3
                else: 
                    tar_idx = 0
                obj_path = ASSET_PATHS[shape]

                bpy.ops.wm.obj_import(filepath=obj_path)
                for idx, obj in enumerate(bpy.context.selected_objects):
                    if idx == tar_idx:
                        continue
                    else:
                        bpy.data.objects.remove(obj, do_unlink=True)

            bpy.ops.object.select_all(action='SELECT')

        for idx, sphere_match in enumerate(sphere_matchs):
            sphere_data_match = re.search(
                r'<([^>]*)>\s*,\s*([0-9.]+)', sphere_match
            )

            if sphere_data_match:
                pos = map(float, sphere_data_match.group(1).split(','))
                pos = self.coord_pov2blend(*pos)
                radius = float(sphere_data_match.group(2))

                obj = bpy.context.selected_objects[idx + 4]
                obj.location = tuple(pos)
                obj.rotation_euler = (math.radians(90), 0, math.radians(90))
                # 如何将物体限制在球体内？

                # 计算物体边界盒的对角线长度（作为"直径"）
                local_bbox_corners = [
                    mathutils.Vector(corner) for corner in obj.bound_box
                ]
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
                safe_scale = scale_factor
                obj.scale = (safe_scale, safe_scale, safe_scale)

                # 将物体位置设回球体中心
                obj.location = tuple(pos)
                if not obj.data.materials:
                    material_name = "MyMaterial"  # 材质名称
                    material = bpy.data.materials.new(name=material_name)
                    # 设置材质的颜色
                    material.diffuse_color = (1.0, 0.0, 0.0, 1.0)
                    obj.data.materials.append(material)

        # mesh_matchs = re.findall(r'object\s*\{\s*cube_mesh[^}]*\}', self.pov_content, re.DOTALL)
        for idx, mesh_match in enumerate(mesh_matchs):
            mesh_data_match = re.search(r'translate\s*<([^>]*)>', mesh_match)
            mesh_scale = re.search(r'scale\s*([0-9.]+)', mesh_match)
            if mesh_data_match:
                origin_pos = list(
                    map(float,
                        mesh_data_match.group(1).split(','))
                )
                pos = self.coord_pov2blend(*pos)
                scale = float(mesh_scale.group(1))

                obj = bpy.context.selected_objects[4 + len(sphere_matchs)
                                                   + idx]
                obj.location = tuple(pos)
                if "Cup" in obj.name:
                    obj.rotation_euler = (math.radians(90), 0, 0)
                elif 'Coffee_cup_withe' in obj.name:
                    obj.rotation_euler = (0, math.radians(90), 0)
                elif 'Book_by_Peter_Iliev_obj' in obj.name:
                    obj.rotation_euler = (math.radians(90), 0, 0)
                elif 'big_pillow' in obj.name:
                    obj.rotation_euler = (math.radians(90), 0, 0)
                elif 'Cylinder' in obj.name:
                    obj.rotation_euler = (0, 0, math.radians(-90))
                else:
                    obj.rotation_euler = (math.radians(90), 0, math.radians(90))

                # 计算物体边界盒的对角线长度（作为"直径"）
                local_bbox_corners = [
                    mathutils.Vector(corner) for corner in obj.bound_box
                ]
                dimensions = [
                    (local_bbox_corners[0]
                     - local_bbox_corners[6]).length,  # z
                    (local_bbox_corners[1]
                     - local_bbox_corners[7]).length,  # x
                    (local_bbox_corners[2] - local_bbox_corners[4]).length  # y
                ]
                max_dimension = np.linalg.norm(dimensions)

                # 计算需要的缩放系数来使物体恰好适合球体
                # 使用直径的一半（半径）与球体半径比较
                scale_factor = scale / max_dimension

                # 应用缩放（略微减小以确保不会超出）
                # 0.95是安全系数，可以根据需要调整
                safe_scale = scale_factor * 1
                obj.scale = (safe_scale, safe_scale, safe_scale)

                # 根据高度调整中心位置
                y_height = dimensions[2] * safe_scale
                shiftted_pos = (origin_pos[0], y_height / 2, origin_pos[2])
                pos = self.coord_pov2blend(*shiftted_pos)

                # 将物体位置设回球体中心
                obj.location = tuple(pos)
                if not obj.data.materials:
                    material_name = "MyMaterial"  # 材质名称
                    material = bpy.data.materials.new(name=material_name)
                    # 设置材质的颜色
                    material.diffuse_color = (1.0, 0.0, 0.0, 1.0)
                    obj.data.materials.append(material)

    def set_camera(self):
        # 设置相机
        camera_match = re.search(
            r'camera\s*{([^}]*)}', self.pov_content, re.DOTALL
        )
        if camera_match:
            camera_text = camera_match.group(1)

            # 提取相机位置
            location_match = re.search(r'location\s*<([^>]*)>', camera_text)
            if location_match:
                x, y, z = map(float, location_match.group(1).split(','))
                x, y, z = self.coord_pov2blend(x, y, z)
                bpy.ops.object.camera_add(location=(x, y, z))
                # 获取刚创建的相机对象并设置为活动相机
                cam = bpy.context.object
                bpy.context.scene.camera = cam

            # 提取相机角度
            angle_match = re.search(r'angle\s*(\d+)', camera_text)
            if angle_match:
                angle = float(angle_match.group(1))
                # 转换 POV-Ray 角度为 Blender 焦距
                bpy.data.cameras['Camera'].lens = 16 / math.tan(
                    math.radians(angle / 2)
                )

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
                constraint.up_axis = 'UP_X'

            # 逆时针旋转相机视角 90 度
            # cam.rotation_euler.z += math.radians(270)  # 绕 Z 轴逆时针旋转 90 度

    def set_light(self):
        # 设置光源
        light_match = re.search(
            r'light_source\s*{([^}]*)}', self.pov_content, re.DOTALL
        )
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
        sphere_matches = re.findall(
            r'sphere\s*{([^}]*)}', self.pov_content, re.DOTALL
        )
        for i, sphere_text in enumerate(sphere_matches):
            # 提取球体位置和半径
            sphere_data_match = re.search(
                r'<([^>]*)>\s*,\s*([0-9.]+)', sphere_text
            )
            if sphere_data_match:
                pos = map(float, sphere_data_match.group(1).split(','))
                pos = self.coord_pov2blend(*pos)
                radius = float(sphere_data_match.group(2))

                # 创建球体
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=radius, location=tuple(pos)
                )
                sphere = bpy.context.object
                sphere.name = f"Sphere_{i}"

                # 提取材质颜色
                color_match = re.search(
                    r'color\s*[a-zA-Z]+|color\s*rgb\s*<([^>]*)>', sphere_text
                )
                if color_match and color_match.group(1):
                    r, g, b = map(float, color_match.group(1).split(','))

                    # 创建材质
                    material = bpy.data.materials.new(
                        name=f"SphereMaterial_{i}"
                    )
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
        sphere_sweep_matches = re.findall(
            r'sphere_sweep\s*{([^}]*)}', self.pov_content, re.DOTALL
        )
        for i, sweep_text in enumerate(sphere_sweep_matches):
            # 提取类型和点数
            header_match = re.search(
                r'(linear_spline|cubic_spline|b_spline)\s*(\d+)', sweep_text
            )
            if header_match:

                # 提取所有点和半径
                points = []
                point_matches = re.findall(
                    r',<([^>]*)>\s*,\s*([0-9.e+-]+)', sweep_text
                )
                for point_match in point_matches:
                    pos = tuple(map(float, point_match[0].split(',')))
                    pos = self.coord_pov2blend(*pos)
                    radius = float(point_match[1])
                    points.append((pos, radius))

                # 创建曲线
                curve_data = bpy.data.curves.new('SweepCurve', 'CURVE')
                curve_data.dimensions = '3D'
                curve_data.resolution_u = 12
                curve_data.bevel_depth = 0.01  #0.01  # 先设置一个小值，后面会根据每个点调整

                # 创建样条
                polyline = curve_data.splines.new('BEZIER')
                polyline.bezier_points.add(len(points) - 1)

                # 设置点位置
                for idx, (pos, radius) in enumerate(points):
                    polyline.bezier_points[idx].co = pos
                    polyline.bezier_points[idx].radius = 2.0
                    polyline.bezier_points[idx].handle_left_type = 'AUTO'
                    polyline.bezier_points[idx].handle_right_type = 'AUTO'

                # 创建曲线对象
                curve_obj = bpy.data.objects.new(
                    'SphereSweep_' + str(i), curve_data
                )
                bpy.context.collection.objects.link(curve_obj)

                # 提取材质颜色
                color_match = re.search(
                    r'pigment\s*{\s*color\s*rgb\s*<([^>]*)>', sweep_text
                )
                if color_match:
                    r, g, b = map(float, color_match.group(1).split(','))

                    # 创建材质
                    material = bpy.data.materials.new(
                        name=f"SweepMaterial_{i}"
                    )
                    material.use_nodes = True

                    nodes = material.node_tree.nodes
                    links = material.node_tree.links

                    # 清除默认节点
                    for node in nodes:
                        nodes.remove(node)

                    # 创建渐变纹理节点
                    gradient_tex = nodes.new(type="ShaderNodeTexGradient")
                    gradient_tex.gradient_type = 'LINEAR'

                    # 创建颜色渐变节点
                    color_ramp = nodes.new(type="ShaderNodeValToRGB")
                    color_ramp.color_ramp.elements[0].color = (
                        0, 0, 0, 1
                    )  # 黑色
                    color_ramp.color_ramp.elements[1].color = (
                        r, g, b, 1
                    )  # 其他颜色

                    # 创建 Principled BSDF 节点
                    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
                    bsdf.inputs["Base Color"].default_value = (r, g, b, 1)

                    # 创建材质输出节点
                    material_output = nodes.new(
                        type="ShaderNodeOutputMaterial"
                    )

                    # 连接节点
                    links.new(color_ramp.inputs[0], gradient_tex.outputs[0])
                    links.new(bsdf.inputs["Base Color"], color_ramp.outputs[0])
                    links.new(
                        material_output.inputs["Surface"], bsdf.outputs[0]
                    )

                    # 将材质应用到曲线对象
                    curve_obj.data.materials.append(material)

                    # 创建一个渐变纹理节点来控制颜色变化
                    gradient_tex = nodes.new(type="ShaderNodeTexCoord")
                    gradient_tex.outputs['Object'].default_value = (0, 0, 0)

                    # 连接渐变纹理节点到颜色渐变节点
                    links.new(
                        color_ramp.inputs[0], gradient_tex.outputs['Object']
                    )

                    # bsdf = material.node_tree.nodes["Principled BSDF"]
                    # bsdf.inputs["Base Color"].default_value = (r, g, b, 1)

                    # 指定材质
                    # curve_obj.data.materials.append(material)

    def render(self):
        bpy.context.scene.camera = bpy.data.objects['Camera']
        bpy.context.scene.frame_set(0)

        bpy.ops.render.render()
        bpy.data.images["Render Result"].save_render(
            f"renders/frame_000000.png"
        )


if __name__ == "__main__":
    # POV 文件路径
    import time
    pov_dir = "/data/wjs/wrp/SoftRoboticaSimulator/work_dirs/povray_continuum_snake/diag"
    pov_file = "/data/wjs/wrp/SoftRoboticaSimulator/work_dirs/povray_continuum_snake/diag/frame_00000.pov"
    # obj_path = "/data/wjs/wrp/SoftRoboticaSimulator/assets/lamed_chair/Lamed_chair.obj"
    # obj_path = "/data/wjs/wrp/SoftRoboticaSimulator/assets/basketball/BasketBall.obj"
    out_dir = "/data/wjs/wrp/SoftRoboticaSimulator/renders"

    pov_settings = BlenderRenderer()
    start_time = time.time()
    # pov_settings.load_pov_settings(pov_file, obj_path)\
    pov_settings.batch_rendering(pov_dir, out_dir)

    s_time = time.time()
    print("render time: ", time.time() - s_time)

    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f}秒")
