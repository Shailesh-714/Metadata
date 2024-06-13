from flask import Flask, request, jsonify
import torch
import cv2
from ultralytics import YOLO
from transformers import pipeline
import numpy as np
from activity_recognition import I3D

app = Flask(__name__)

# Load models
object_detection_model = YOLO('yolov5s.pt')
activity_recognition_model = I3D(num_classes=10)
activity_recognition_model.load_state_dict(torch.load('i3d_activity_recognition.pth', map_location=torch.device('cpu')))
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

@app.route('/')
def home():
    return "Welcome to the Activity Recognition API!"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detect_objects(image)
    events = detect_events([detections])
    summary = summarize_events(events)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
