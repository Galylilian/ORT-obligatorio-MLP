from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def predict_yolo(image_path):
    results = model(image_path)
    detections = results[0].boxes

    return 1 if detections and len(detections) > 0 else 0