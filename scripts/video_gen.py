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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for filename in filenames:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
            
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))
            
        video_writer.write(img)

    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description="Convert an episode's map images into a video.")
    parser.add_argument("input_folder")
    parser.add_argument("output_video")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    pic_to_video(args.input_folder, args.output_video, fps=args.fps)
