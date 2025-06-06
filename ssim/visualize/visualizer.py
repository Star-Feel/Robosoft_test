import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import elastica as ea


def plot_video(
    plot_params: dict,
    sphere_params: dict,
    sphere,
    video_name="video.mp4",
    fps=15,
    xlim=(0, 4),
    ylim=(-1, 1),
):  # (time step, x/y/z, node)

    positions_over_time = np.array(plot_params["position"])
    cylinder_positions_over_time = np.array(sphere_params["position"])

    print("plot video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(
        title="Movie Test", artist="Matplotlib", comment="Movie support!"
    )
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("z [m]", fontsize=16)
    ax.set_ylabel("x [m]", fontsize=16)
    rod_lines_2d = ax.plot(
        positions_over_time[0][2], positions_over_time[0][0]
    )[0]

    # 初始化一个变量来保存当前的圆形对象
    current_circle = None

    with writer.saving(fig, video_name, dpi=150):
        for time in tqdm(range(1, len(plot_params["time"]))):
            # 更新杆的位置
            rod_lines_2d.set_xdata(positions_over_time[time][2])
            rod_lines_2d.set_ydata(positions_over_time[time][0])

            # 移除旧的圆形（如果存在）
            if current_circle is not None:
                current_circle.remove()

            # 添加新的圆形
            center_x = cylinder_positions_over_time[time][2]
            center_y = cylinder_positions_over_time[time][0]
            radius = sphere.radius[0]
            current_circle = plt.Circle((center_x, center_y),
                                        radius,
                                        edgecolor='b',
                                        facecolor='lightblue')
            ax.add_patch(current_circle)

            # 捕捉当前帧
            writer.grab_frame()

    # 关闭图形
    plt.close(plt.gcf())


# def create_3d_animation(position_rod: np.ndarray,
#                         position_sphere: np.ndarray,
#                         sphere_radius: float,
#                         save_path=None,
#                         fps=30,
#                         skip=1):
#     """
#     Create and save a 3D animation of the rod motion with sphere
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # 合并杆和球体的坐标范围（包含球体边界）
#     sphere_min = position_sphere - sphere_radius
#     sphere_max = position_sphere + sphere_radius
#     all_positions = np.concatenate([position_rod, sphere_min, sphere_max],
#                                    axis=-1)

#     # 计算动态坐标范围
#     x_min, x_max = np.min(all_positions[:, 0, :]), np.max(all_positions[:,
#                                                                         0, :])
#     y_min, y_max = np.min(all_positions[:, 1, :]), np.max(all_positions[:,
#                                                                         1, :])
#     z_min, z_max = np.min(all_positions[:, 2, :]), np.max(all_positions[:,
#                                                                         2, :])

#     # 添加边距
#     margin = 0.1
#     x_min -= margin * (x_max - x_min)
#     x_max += margin * (x_max - x_min)
#     y_min -= margin * (y_max - y_min)
#     y_max += margin * (y_max - y_min)
#     z_min -= margin * (z_max - z_min)
#     z_max += margin * (z_max - z_min)

#     # 设置动态坐标轴
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
#     ax.set_zlim(z_min, z_max)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Rod Motion Animation')

#     # 初始化杆的曲线
#     line, = ax.plot([], [], [], 'r-', lw=2)

#     # 初始化球体参数
#     u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
#     x = sphere_radius * np.cos(u) * np.sin(v)
#     y = sphere_radius * np.sin(u) * np.sin(v)
#     z = sphere_radius * np.cos(v)
#     sphere_surface = ax.plot_surface(x, y, z, color='b', alpha=0.6)

#     # 时间显示
#     time_template = 'Time: %.3fs'
#     time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
#     num_frames = position_rod.shape[0]

#     def init():
#         line.set_data([], [])
#         line.set_3d_properties([])

#         time_text.set_text('')
#         return line, sphere_surface, time_text

#     def update(frame):
#         # 更新杆的位置
#         line.set_data(position_rod[frame, 0, :], position_rod[frame, 1, :])
#         line.set_3d_properties(position_rod[frame, 2, :])

#         # 更新球体位置
#         cx, cy, cz = position_sphere[frame]
#         nonlocal sphere_surface
#         sphere_surface.remove()
#         sphere_surface = ax.plot_surface(x + cx,
#                                          y + cy,
#                                          z + cz,
#                                          color='b',
#                                          alpha=0.6)

#         # 更新时间显示
#         time_text.set_text(time_template % (frame / fps))
#         return line, sphere_surface, time_text

#     progress_callback = tqdm(total=num_frames // skip,
#                              desc="Generating frames")

#     def update_with_progress(frame):
#         result = update(frame)
#         progress_callback.update(1)
#         return result

#     anim = animation.FuncAnimation(fig,
#                                    update_with_progress,
#                                    frames=range(0, num_frames, skip),
#                                    init_func=init,
#                                    blit=True,
#                                    interval=1000 / fps)

#     # 保存视频
#     if save_path:
#         writer = animation.FFMpegWriter(fps=fps,
#                                         metadata={'artist': 'DeepSeek'},
#                                         bitrate=5000)
#         anim.save(save_path, writer=writer)

#     plt.close()
#     return anim


def create_3d_animation(
    position_rod: np.ndarray,
    position_sphere: np.ndarray,  # 新增球体位置参数 (time_steps, 3)
    sphere_radius: float,  # 新增球体半径参数
    save_path=None,
    fps=30,
    skip=1
):
    """
    Create and save a 3D animation of the rod motion with sphere

    Args:
        position_rod: 杆的位置数据 (time_steps, 3, n_elems)
        position_sphere: 球体中心位置数据 (time_steps, 3)
        sphere_radius: 球体半径
        save_path: 视频保存路径
        fps: 帧率
        skip: 跳帧参数（优化性能）
    """
    # 初始化图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 合并杆和球体的坐标范围
    all_positions = np.concatenate([position_rod, position_sphere], axis=-1)

    # 计算统一的坐标范围
    x_min, x_max = np.min(all_positions[:,
                                        0, :]), np.max(all_positions[:, 0, :])
    y_min, y_max = np.min(all_positions[:,
                                        1, :]), np.max(all_positions[:, 1, :])
    z_min, z_max = np.min(all_positions[:,
                                        2, :]), np.max(all_positions[:, 2, :])

    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    z_min -= margin * z_range
    z_max += margin * z_range

    # Set axis limits and labels
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rod Motion Animation')

    # 初始化杆的曲线
    line, = ax.plot([], [], [], 'r-', lw=2)

    # 初始化球体（使用meshgrid创建球面）
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = sphere_radius * np.outer(np.cos(u), np.sin(v))
    y = sphere_radius * np.outer(np.sin(u), np.sin(v))
    z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    global sphere_surface
    sphere_surface = ax.plot_surface(x, y, z, color='b', alpha=0.6)

    # 时间显示
    time_template = 'Time: %.3fs'
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    num_frames = all_positions.shape[0]

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        # 更新杆的位置
        line.set_data(position_rod[frame, 0, :], position_rod[frame, 1, :])
        line.set_3d_properties(position_rod[frame, 2, :])

        # 更新球体位置
        cx, cy, cz = position_sphere[frame]
        x_new = x + cx
        y_new = y + cy
        z_new = z + cz
        # sphere_surface._verts3d = (x + cx[0], y + cy[0], z + cz[0])
        global sphere_surface
        sphere_surface.remove()  # 删除旧对象
        sphere_surface = ax.plot_surface(
            x_new, y_new, z_new, color='b', alpha=0.6
        )
        # 更新时间显示
        time_text.set_text(time_template % (frame / fps))

        return line, sphere_surface, time_text

    progress_callback = tqdm(
        total=num_frames // skip, desc="Generating frames"
    )

    def update_with_progress(frame):
        result = update(frame)
        progress_callback.update(1)
        return result

    anim = animation.FuncAnimation(
        fig,
        update_with_progress,
        frames=range(0, num_frames, skip),
        init_func=init,
        blit=True,
        interval=1000 / fps
    )

    # 保存视频
    if save_path:
        writer = animation.FFMpegWriter(
            fps=fps, metadata={'artist': 'DeepSeek'}, bitrate=5000
        )
        anim.save(save_path, writer=writer)

    plt.close()
    return anim


def rod_objects_3d_visualize(
    position_rod: np.ndarray,
    position_objects: list[np.ndarray],
    objects: list,  # 新增球体半径参数
    save_path=None,
    fps=30,
    skip=1,
    xlim=None,
    ylim=None,
    zlim=None,
):
    """
    Create and save a 3D animation of the rod motion with sphere

    Args:
        position_rod: 杆的位置数据 (time_steps, 3, n_elems)
        position_sphere: 球体中心位置数据 (time_steps, 3)
        sphere_radius: 球体半径
        save_path: 视频保存路径
        fps: 帧率
        skip: 跳帧参数（优化性能）
    """
    # 初始化图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 合并杆和球体的坐标范围
    all_positions = np.concatenate([position_rod, *position_objects], axis=-1)

    # 计算统一的坐标范围
    x_min, x_max = np.min(all_positions[:,
                                        0, :]), np.max(all_positions[:, 0, :])
    y_min, y_max = np.min(all_positions[:,
                                        1, :]), np.max(all_positions[:, 1, :])
    z_min, z_max = np.min(all_positions[:,
                                        2, :]), np.max(all_positions[:, 2, :])

    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    z_min -= margin * z_range
    z_max += margin * z_range

    # Set axis limits and labels
    ax.set_xlim(xlim if xlim is not None else (x_min, x_max))
    ax.set_ylim(ylim if ylim is not None else (y_min, y_max))
    ax.set_zlim(zlim if zlim is not None else (z_min, z_max))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rod Motion Animation')

    # 初始化杆的曲线
    line, = ax.plot([], [], [], 'r-', lw=2)
    # 初始化objects
    objects_plot = []
    for obj in objects:
        if isinstance(obj, ea.Sphere):
            # 初始化球体（使用meshgrid创建球面）
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = obj.radius * np.outer(np.cos(u), np.sin(v))
            y = obj.radius * np.outer(np.sin(u), np.sin(v))
            z = obj.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            sphere_surface = ax.plot_surface(x, y, z, color='b', alpha=0.6)
            objects_plot.append(sphere_surface)

    # 时间显示
    time_template = 'Time: %.3fs'
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    num_frames = all_positions.shape[0]

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        # 更新杆的位置
        line.set_data(position_rod[frame, 0, :], position_rod[frame, 1, :])
        line.set_3d_properties(position_rod[frame, 2, :])

        nonlocal objects_plot
        for plot in objects_plot:
            plot.remove()
        objects_plot = []
        # 更新球体位
        for idx, obj in enumerate(objects):
            if isinstance(obj, ea.Sphere):

                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = obj.radius * np.outer(np.cos(u), np.sin(v))
                y = obj.radius * np.outer(np.sin(u), np.sin(v))
                z = obj.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                cx, cy, cz = position_objects[idx][frame]
                x_new = x + cx
                y_new = y + cy
                z_new = z + cz
                sphere_surface = ax.plot_surface(
                    x_new, y_new, z_new, color='b', alpha=0.6
                )
                objects_plot.append(sphere_surface)

        # 更新时间显示
        time_text.set_text(time_template % (frame / fps))

        return line, time_text

    progress_callback = tqdm(
        total=num_frames // skip, desc="Generating frames"
    )

    def update_with_progress(frame):
        result = update(frame)
        progress_callback.update(1)
        return result

    anim = animation.FuncAnimation(
        fig,
        update_with_progress,
        frames=range(0, num_frames, skip),
        init_func=init,
        blit=True,
        interval=1000 / fps
    )

    # 保存视频
    if save_path:
        writer = animation.FFMpegWriter(
            fps=fps, metadata={'artist': 'DeepSeek'}, bitrate=5000
        )
        anim.save(save_path, writer=writer)

    plt.close()
    return anim


def plot_contour(
    positions: np.ndarray,
    xlim=None,
    ylim=None,
    levels=50,
    save_path=None,
):
    """
    Plot a simple 2D line plot based on positions.

    Args:
        positions: 2D coordinates of shape (time_steps, n_points, 2).
        xlim: Tuple specifying x-axis limits (optional).
        ylim: Tuple specifying y-axis limits (optional).
    """
    # Extract x and y coordinates
    x = positions[:, :, 0]
    y = positions[:, :, 1]

    # Plot the positions as a simple line plot
    plt.figure(figsize=(8, 6))
    for i in range(x.shape[1]):
        plt.plot(x[:, i], y[:, i], label=f"Point {i + 1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Trajectory Plot")
    # plt.legend()

    # Set axis limits if provided
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_contour_3d(
    positions: np.ndarray,
    xlim=None,
    ylim=None,
    zlim=None,
    view_angle=(30, 30),
    save_path=None,
    figsize=(10, 8),
):
    """
    Plot a 3D trajectory based on positions.

    Args:
        positions: 3D coordinates of shape (time_steps, n_points, 3).
        xlim: Tuple specifying x-axis limits (optional).
        ylim: Tuple specifying y-axis limits (optional).
        zlim: Tuple specifying z-axis limits (optional).
        view_angle: Tuple (elevation, azimuth) for the viewing angle (optional).
        save_path: Path to save the figure (optional).
        figsize: Figure size as (width, height) in inches (optional).
    """
    # Extract x, y and z coordinates
    x = positions[:, :, 0]
    y = positions[:, :, 1]
    z = positions[:, :, 2]

    # Create a 3D figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the positions as 3D trajectories
    for i in range(x.shape[1]):
        ax.plot(x[:, i], y[:, i], z[:, i], label=f"Point {i + 1}")

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory Plot")

    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

    # Add a legend if there are multiple points
    if x.shape[1] > 1:
        ax.legend()

    # Improve the aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Add grid for better visibility
    ax.grid(True)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_contour_with_spheres(
    positions: np.ndarray,
    spheres: list,
    xlim=None,
    ylim=None,
    save_path=None,
):
    """
    Plot a 2D line plot based on positions and overlay spheres.

    Args:
        positions: 2D coordinates of shape (time_steps, n_points, 2).
        spheres: Array of spheres, each defined as (x, y, z, r).
        xlim: Tuple specifying x-axis limits (optional).
        ylim: Tuple specifying y-axis limits (optional).
    """
    # Extract x and z coordinates
    x = positions[:, :, 0]
    z = positions[:, :, 1]

    # Plot the positions as a simple line plot
    plt.figure(figsize=(8, 6))
    for i in range(x.shape[1]):
        plt.plot(z[:, i], x[:, i], label=f"Point {i + 1}")
    plt.xlabel("Z")
    plt.ylabel("X")
    plt.title("2D Trajectory Plot")

    # Overlay spheres
    for sphere in spheres:
        center_x, _, center_z, radius = sphere
        circle = plt.Circle((center_z, center_x), radius, color='b', alpha=0.3)
        plt.gca().add_patch(circle)

    # Set axis limits if provided
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_contour_with_spheres_3d(
    positions: np.ndarray,
    spheres: list,
    xlim=None,
    ylim=None,
    zlim=None,
    view_angle=(30, 30),
    sphere_alpha=0.3,
    sphere_color='royalblue',
    sphere_resolution=20,
    save_path=None,
    figsize=(10, 8),
):
    """
    Plot a 3D trajectory based on positions and overlay spheres.

    Args:
        positions: 3D coordinates of shape (time_steps, n_points, 3).
        spheres: List of spheres, each defined as (x, y, z, r).
        xlim: Tuple specifying x-axis limits (optional).
        ylim: Tuple specifying y-axis limits (optional).
        zlim: Tuple specifying z-axis limits (optional).
        view_angle: Tuple (elevation, azimuth) for the viewing angle (optional).
        sphere_alpha: Transparency of spheres (0-1).
        sphere_color: Color of the spheres.
        sphere_resolution: Resolution of the sphere meshes.
        save_path: Path to save the figure (optional).
        figsize: Figure size as (width, height) in inches (optional).
    """
    # Extract x, y and z coordinates
    x = positions[:, :, 0]
    y = positions[:, :, 1]
    z = positions[:, :, 2]

    # Create a 3D figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the positions as 3D trajectories
    for i in range(x.shape[1]):
        ax.plot(x[:, i], y[:, i], z[:, i], label=f"Point {i + 1}")

    # Add spheres
    for sphere in spheres:
        center_x, center_y, center_z, radius = sphere

        # Create a sphere
        u = np.linspace(0, 2 * np.pi, sphere_resolution)
        v = np.linspace(0, np.pi, sphere_resolution)

        sphere_x = center_x + radius * np.outer(np.cos(u), np.sin(v))
        sphere_y = center_y + radius * np.outer(np.sin(u), np.sin(v))
        sphere_z = center_z + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the sphere
        ax.plot_surface(
            sphere_x,
            sphere_y,
            sphere_z,
            color=sphere_color,
            alpha=sphere_alpha,
            linewidth=0,
            antialiased=True
        )

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory with Spheres")

    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

    # Add a legend if there are multiple points
    if x.shape[1] > 1:
        ax.legend()

    # Improve the aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Add grid for better visibility
    ax.grid(True)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    return fig, ax
