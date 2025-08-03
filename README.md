# 🚘 AutoSight AI – Real-Time Autonomous Driving Perception System

**AutoSight AI** is a real-time AI-based system for object detection, depth estimation, and dynamic risk assessment — built to simulate core perception capabilities in autonomous vehicles using your webcam or dashcam footage.

![AutoSight Demo](https://github.com/Adityajl/AutoSightAI/blob/master/Screenshot%20From%202025-08-03%2021-02-54.png) <!-- Replace with your GIF -->

---

## 🧠 What It Does

- 🚗 Detects road objects using YOLOv8
- 📏 Estimates depth using MiDaS
- ⚠️ Assesses risk levels (High / Medium / Low)
- 🖥️ Displays annotated frames with real-time overlays

---

## 🛠 Tech Stack

| Component        | Tech                         |
|------------------|------------------------------|
| Detection        | YOLOv8                       |
| Depth Estimation | MiDaS (Intel ISL)            |
| Risk Logic       | Custom Rules in Python       |
| Interface        | Streamlit + OpenCV           |
| Language         | Python 3.11+                 |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone git@github.com:Adityajl/AutoSightAI.git
cd AutoSightAI

# 2. Create virtual env and activate
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run webcam live detection
python main.py

# 5. Or run Streamlit UI for video upload
streamlit run app.py
