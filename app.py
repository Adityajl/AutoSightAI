# app.py
import streamlit as st
import tempfile
import cv2
import os
from perception.detector import ObjectDetector
from perception.depth_estimator import DepthEstimator
from utils.risk_assessment import assess_risk

st.set_page_config(page_title="AutoSight AI", layout="wide")

st.title("🚘 AutoSight AI – Smart Perception for Autonomous Driving")
st.markdown("Upload a driving video to detect objects, estimate depth, and assess risk in real-time.")

uploaded_file = st.file_uploader("📂 Upload a driving scene video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_vid = tempfile.NamedTemporaryFile(delete=False)
    temp_vid.write(uploaded_file.read())
    temp_vid_path = temp_vid.name

    detector = ObjectDetector("yolov8n.pt")
    depth_model = DepthEstimator()

    stframe = st.empty()
    cap = cv2.VideoCapture(temp_vid_path)

    while cap.isOpened():
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
                est_meters = est_depth * 10
                distance_text = f"{est_meters:.2f}m"
            except:
                est_meters = None
                distance_text = "N/A"

            risk = assess_risk(label, est_meters)
            if risk == "High":
                color = (0, 0, 255)
            elif risk == "Medium":
                color = (0, 255, 255)
            elif risk == "Low":
                color = (0, 255, 0)
            else:
                color = (180, 180, 180)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{distance_text} | {risk}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        combined = cv2.hconcat([frame, depth_vis])
        combined = cv2.resize(combined, (1280, 480))
        stframe.image(combined, channels="BGR")

