import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm


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
    metadata = dict(title="Movie Test",
                    artist="Matplotlib",
                    comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("z [m]", fontsize=16)
    ax.set_ylabel("x [m]", fontsize=16)
    rod_lines_2d = ax.plot(positions_over_time[0][2],
                           positions_over_time[0][0])[0]

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
        skip=1):
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
    x_min, x_max = np.min(all_positions[:, 0, :]), np.max(all_positions[:,
                                                                        0, :])
    y_min, y_max = np.min(all_positions[:, 1, :]), np.max(all_positions[:,
                                                                        1, :])
    z_min, z_max = np.min(all_positions[:, 2, :]), np.max(all_positions[:,
                                                                        2, :])

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
        sphere_surface = ax.plot_surface(x_new,
                                         y_new,
                                         z_new,
                                         color='b',
                                         alpha=0.6)
        # 更新时间显示
        time_text.set_text(time_template % (frame / fps))

        return line, sphere_surface, time_text

    progress_callback = tqdm(total=num_frames // skip,
                             desc="Generating frames")

    def update_with_progress(frame):
        result = update(frame)
        progress_callback.update(1)
        return result

    anim = animation.FuncAnimation(fig,
                                   update_with_progress,
                                   frames=range(0, num_frames, skip),
                                   init_func=init,
                                   blit=True,
                                   interval=1000 / fps)

    # 保存视频
    if save_path:
        writer = animation.FFMpegWriter(fps=fps,
                                        metadata={'artist': 'DeepSeek'},
                                        bitrate=5000)
        anim.save(save_path, writer=writer)

    plt.close()
    return anim
