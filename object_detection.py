import torch
import cv2
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detect_objects(frame):
    results = model(frame)
    labels = results.names
    detected = []
    for result in results.pred:
        detected.extend([labels[int(cls)] for cls in result[:, -1].cpu().numpy()])
    return detected

# Example usage
cap = cv2.VideoCapture("cctv_footage.mp4")
all_detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    detections = detect_objects(frame)
    all_detections.append(detections)

cap.release()
