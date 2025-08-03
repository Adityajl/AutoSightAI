# main.py
import cv2
from perception.detector import ObjectDetector
from perception.depth_estimator import DepthEstimator
from utils.risk_assessment import assess_risk

def main():
    detector = ObjectDetector("yolov8n.pt")
    depth_model = DepthEstimator()

    cap = cv2.VideoCapture(0)  # Use 0 for webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        depth_map, depth_vis = depth_model.estimate(frame)

        for det in detections:
            label = det["label"]
            conf = det["conf"]
            x1, y1, x2, y2 = det["bbox"]
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            try:
                est_depth = depth_map[center_y, center_x]
                est_meters = est_depth * 10  # MiDaS returns relative depth; scale it
                distance_text = f"{est_meters:.2f}m"
            except:
                est_meters = None
                distance_text = "N/A"

            # Risk level assessment
            risk = assess_risk(label, est_meters)
            if risk == "High":
                color = (0, 0, 255)
            elif risk == "Medium":
                color = (0, 255, 255)
            elif risk == "Low":
                color = (0, 255, 0)
            else:
                color = (180, 180, 180)

            # Draw detection and info
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{distance_text} | {risk}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        combined = cv2.hconcat([frame, depth_vis])
        cv2.imshow("AutoSight AI – Detection + Depth + Risk", combined)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
