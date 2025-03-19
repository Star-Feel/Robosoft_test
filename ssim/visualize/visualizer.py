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


def create_3d_animation(
        position_rod,
        position_sphere,  # 新增球体位置参数 (time_steps, 3)
        sphere_radius,  # 新增球体半径参数
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
    all_positions = np.concatenate(
        [position_rod.reshape(-1, 3),
         position_sphere.reshape(-1, 3)], axis=0)

    # 计算统一的坐标范围
    mins = all_positions.min(axis=0)
    maxs = all_positions.max(axis=0)
    ranges = maxs - mins
    margin = 0.1
    ax.set_xlim(mins[0] - margin * ranges[0], maxs[0] + margin * ranges[0])
    ax.set_ylim(mins[1] - margin * ranges[1], maxs[1] + margin * ranges[1])
    ax.set_zlim(mins[2] - margin * ranges[2], maxs[2] + margin * ranges[2])

    # 初始化杆的曲线
    rod_line, = ax.plot([], [], [], 'r-', lw=2)

    # 初始化球体（使用meshgrid创建球面）
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = sphere_radius * np.outer(np.cos(u), np.sin(v))
    y = sphere_radius * np.outer(np.sin(u), np.sin(v))
    z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    sphere_surface = ax.plot_surface(x, y, z, color='b', alpha=0.6)

    # 时间显示
    time_template = 'Time: %.3fs'
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def update(frame):
        # 更新杆的位置
        rod_line.set_data(position_rod[frame, 0, :], position_rod[frame, 1, :])
        rod_line.set_3d_properties(position_rod[frame, 2, :])

        # 更新球体位置
        cx, cy, cz = position_sphere[frame]
        sphere_surface._verts3d = (x + cx, y + cy, z + cz)

        # 更新时间显示
        time_text.set_text(time_template % (frame / fps))

        return rod_line, sphere_surface, time_text

    # 创建动画
    anim = animation.FuncAnimation(fig,
                                   update,
                                   frames=range(0, position_rod.shape[0],
                                                skip),
                                   interval=1000 / fps,
                                   blit=False)

    # 保存视频
    if save_path:
        writer = animation.FFMpegWriter(fps=fps,
                                        metadata={'artist': 'DeepSeek'},
                                        bitrate=5000)
        anim.save(save_path, writer=writer)

    plt.close()
    return anim
