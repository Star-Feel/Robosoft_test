import os
import moviepy
from moviepy import VideoFileClip

def convert_videos_to_gifs():
    # 获取当前路径下的所有文件
    files = os.listdir('.')
    # 筛选出所有 .mp4 文件
    mp4_files = [f for f in files if f.endswith('.mp4')]

    for mp4_file in mp4_files:
        # 加载视频文件
        with VideoFileClip(mp4_file) as video:
            # 设置输出gif文件名
            gif_file = mp4_file.replace('.mp4', '.gif')
            # 转换为gif
            video.write_gif(gif_file)
            print(f'Converted {mp4_file} to {gif_file}')

if __name__ == '__main__':
    convert_videos_to_gifs()
