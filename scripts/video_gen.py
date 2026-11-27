import cv2
import os
import numpy as np

def pic_to_video(input_folder, output_path, fps=24, img_size=None):
    filenames = sorted([f for f in os.listdir(input_folder)],
                      key=lambda x: x.lower())
    
    first_img = cv2.imread(os.path.join(input_folder, filenames[0]))
    if img_size is None:
        height, width, _ = first_img.shape
    else:
        width, height = img_size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for filename in filenames:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
            
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))
            
        video_writer.write(img)

    video_writer.release()
    print(f"视频已保存至：{output_path}")

# 使用示例
if __name__ == "__main__":
    
    input_folder = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouqihang-240108100050/HuNav/observations/20250627_153846/episode_1/maps"  # 替换为你的图片文件夹路径
    output_video = "test_map.mp4"         # 输出视频路径

    pic_to_video(input_folder, output_video, fps=10)  # 设置为30帧/秒
