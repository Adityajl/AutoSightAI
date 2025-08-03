# perception/detector.py
from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect(self, frame):
        results = self.model(frame, stream=True)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": self.model.names[cls],
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections
