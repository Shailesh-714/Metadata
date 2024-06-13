import cv2
import torch
from ultralytics import YOLO
from transformers import pipeline
import numpy as np

from activity_recognition import I3D
from event_detection import detect_events
from summarization import summarize_events

# Load YOLOv5 model
object_detection_model = YOLO('yolov5su.pt')

# Load I3D model
activity_recognition_model = I3D(num_classes=10)
activity_recognition_model.load_state_dict(torch.load('i3d_activity_recognition.pth'))
activity_recognition_model.eval()

# Load summarizer
summarizer = pipeline("summarization")

def detect_objects(frame):
    results = object_detection_model(frame)
    detected = []
    for result in results:
        labels = result.names
        for cls in result.boxes.cls:
            detected.append(labels[int(cls)])
    return detected

def detect_events(detections):
    events = []
    for detection in detections:
        if 'person' in detection and 'car' in detection:
            events.append('Person near a car')
    return events

def summarize_events(events):
    text = " ".join(events)
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Process video
cap = cv2.VideoCapture("cctv_footage.mp4")
all_detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    detections = detect_objects(frame)
    all_detections.append(detections)

cap.release()

events = detect_events(all_detections)
summary = summarize_events(events)
print(summary)
