from .povray import Stages, pyelastica_rod, render
from functools import partial
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import os
import numpy as np


class POVRayRenderer:

    def __init__(
        self,
        output_filename="povray_video",
        output_images_dir="povray_frames",
        fps=20.0,
        width=1920,
        height=1080,
        display_frames="Off",
        included: list[str] = [
            "ssim/visualize/povray/includes/default.inc",
            "ssim/visualize/povray/includes/meshes.inc"
        ],
        top_camera_position=[0, 10, 2],
        top_camera_look_at=[0, 0, 2],
    ):
        self._ouput_filename = output_filename
        self._output_images_dir = output_images_dir
        self._fps = fps
        self._width = width
        self._height = height
        # Display povray images during the rendering. ['On', 'Off']
        self._display_frames = display_frames
        self._stages = Stages()

        # Externally Including Files (USER DEFINE)
        # If user wants to include other POVray objects such as grid or coordinate axes,
        # objects can be defined externally and included separately.
        self._included = included

        self._current_frame = 0
        self._batch = []

        self.reset_stage(top_camera_position, top_camera_look_at)

    def reset_stage(self, top_camera_position=[0, 10, 2], top_camera_look_at=[0, 0, 2]):
        stages = Stages()
        stages.add_camera(
            # Add first-person viewpoint
            location=[0, 0, 2],
            angle=30,
            look_at=[0.0, 0, 0],
            name="fpv",
        )
        stages.add_camera(
            # Add top viewpoint
            location=top_camera_position,
            angle=30,
            look_at=top_camera_look_at,
            sky=[-1, 0, 0],
            name="top",
        )
        stages.add_camera(
            # Add diagonal viewpoint
            location=[7.0, 4, 2],
            angle=30,
            look_at=[0.0, 0, 2],
            name="diag",
        )
        stages.add_light(
            # Sun light
            position=[1500, 2500, -1000],
            color="White",
            camera_name="All",
        )

        self._stages = stages

    def add_stage_object(self, object_type, name, **kwargs):
        assert object_type in ["box", "sphere", "mesh"], "Invalid object type"
        self._stages.add_stage_object(object_type, name, **kwargs)

    def fpv_update(self, xs, base_radius):
        # xs: (3, num_element)
        # base_radius: (num_element,)
        fpv_camera = self._stages.cameras['fpv']
        pre_head, head = xs[:, -2], xs[:, -1]
        head_radius = base_radius[-1]
        direction = (head - pre_head) / np.linalg.norm(head - pre_head)

        fpv_camera.update(
            location=head + direction * head_radius,
            look_at=head + direction * head_radius * 2,
        )

    def render_single_step(self, data: dict, save_img=False) -> int:
        # Convert data to numpy array
        xs = np.array(data["rod_position"])  # shape: (3, num_element)

        rod_radius = np.array(data["rod_radius"])
        # If the data contains multiple rod, this part can be modified to include
        # multiple rods.
        rod_object = pyelastica_rod(
            x=xs,
            r=rod_radius,
            color="rgb<0.45,0.39,1>",
        )

        self.fpv_update(xs, rod_radius)
        _stage_scripts = self._stages.generate_scripts()

        # Make Directory for each camera
        for view_name, stage_script in _stage_scripts.items():
            output_path = os.path.join(self._output_images_dir, view_name)
            os.makedirs(output_path, exist_ok=True)

            # Collect povray scripts
            script = []
            script.extend(['#include "{}"'.format(s) for s in self._included])
            script.append(stage_script)

            script.append(rod_object)
            pov_script = "\n".join(script)

            # Write .pov script file
            file_path = os.path.join(
                output_path, "frame_{:05d}".format(self._current_frame))
            with open(file_path + ".pov", "w+") as f:
                f.write(pov_script)
            self._batch.append(file_path)

        if save_img:
            self.process_povray(self._batch[-2:])
        self._current_frame += 1
        return self._current_frame - 1

    def process_povray(
        self,
        file_paths=None,
        multi_processing=False,
        thread_per_agent=4,
        bar=True,
    ):
        # Process POVray
        # For each frames, a 'png' image file is generated in self._output_images_dir directory.

        if file_paths is None:
            file_paths = self._batch
        if bar:
            pbar = tqdm(total=len(file_paths), desc="Rendering .png")
        if multi_processing:
            num_agent = 16
            func = partial(
                render,
                width=self._width,
                height=self._height,
                display=self._display_frames,
                pov_thread=1,
            )
            with Pool(num_agent) as p:
                for _ in p.imap_unordered(func, file_paths):
                    # (TODO) POVray error within child process could be an issue
                    if bar:
                        pbar.update()
        else:
            for filename in self._batch:
                render(
                    filename,
                    width=self._width,
                    height=self._height,
                    display=self._display_frames,
                    pov_thread=multiprocessing.cpu_count(),
                )
                if bar:
                    pbar.update()

    def create_video(self, only_top=False):
        _stage_scripts = self._stages.generate_scripts()
        if only_top:
            view_name = "top"
            imageset_path = os.path.join(self._output_images_dir, view_name)

            filename = self._ouput_filename + "_" + view_name + ".mp4"

            os.system(
                f"ffmpeg -r {self._fps} -i {imageset_path}/frame_%05d.png videos/{filename}"
            )
            return 

        for view_name in _stage_scripts.keys():
            imageset_path = os.path.join(self._output_images_dir, view_name)

            filename = self._ouput_filename + "_" + view_name + ".mp4"

            os.system(
                f"ffmpeg -r {self._fps} -i {imageset_path}/frame_%05d.png {filename} -y"
            )
