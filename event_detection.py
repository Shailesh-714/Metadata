def detect_events(detections):
    events = []
    for detection in detections:
        if 'person' in detection and 'car' in detection:
            events.append('Person near a car')
    return events

# Example usage
example_detections = [['person', 'car'], ['person'], ['car'], ['person', 'bicycle']]
events = detect_events(example_detections)
print(events)
