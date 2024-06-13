import cv2
import os
import numpy as np

def extract_frames(video_path, output_dir, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    count = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
    cap.release()

def data_augmentation(image_dir, output_dir):
    # Implement your augmentation logic here
    pass


extract_frames('cctv_footage.mp4', 'frames')
data_augmentation('frames', 'augmented_frames')
