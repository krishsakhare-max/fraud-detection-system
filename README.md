# AI-Powered Fraud Detection Dashboard

Streamlit dashboard for live or recorded video monitoring in exam-proctoring and remote-surveillance scenarios.

The app combines face detection, mobile phone detection, head-pose checks, fraud scoring, event logging, and a final session summary.

## What This Submission Covers

### Detection Modules

- Face detection
- Face verification hook for future extension
- Object detection for mobile phones
- Fraud event logging

### Dashboard

- Live webcam analysis
- Recorded video analysis
- Fraud alerts with timestamps
- Final session fraud summary

### Documentation

- Approach
- Dataset and resources used
- Model comparison
- Challenges faced
- Future improvements

## Project Structure

```text
fraud_detection/
├── app.py
├── modules/
│   ├── face_detection.py
│   ├── face_verify.py
│   ├── fraud_score.py
│   ├── head_pose.py
│   └── phone_detector.py
├── utils/
│   └── logger.py
├── logs/
├── registered_faces/
├── requirements.txt
└── README.md
```

## Core Features

- Live camera monitoring with a stable preview canvas
- Uploaded video support for recorded analysis
- Phone detection using a pre-trained YOLOv8 model
- Face-count analysis using OpenCV Haar cascades
- Head-pose estimation using MediaPipe FaceMesh when available
- Fraud score levels: `LOW`, `MEDIUM`, `HIGH`
- Fraud event CSV logging with timestamps
- Session summary with total events and risk duration

## Face Verification Note

The repository includes a lightweight `modules/face_verify.py` compatibility layer so the submission stays runnable in environments where the heavier identity-recognition dependency is unavailable.

If you want to extend the project later, you can swap that stub for a real verification model without changing the dashboard flow.

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the dashboard

```powershell
streamlit run app.py
```

## How To Use

1. Open the sidebar.
2. Choose `Webcam` or `Uploaded video`.
3. Click `Start Session`.
4. Watch the live frame, alerts, log table, and summary update.
5. Click `Stop Session` when finished.

## Approach

1. Capture a frame from webcam or uploaded video.
2. Normalize the frame to a fixed canvas for stable rendering.
3. Run:
   - face detection
   - mobile phone detection
   - head-pose estimation
4. Combine the results into a fraud score.
5. Log medium/high-risk events to CSV.
6. Display alerts and the final session summary in Streamlit.

## Dataset and Resources Used

This project does not train a new model from scratch.

Resources used:

- OpenCV Haar cascade for face detection
- YOLOv8 COCO weights (`yolov8n.pt`) for mobile phone detection
- MediaPipe FaceMesh landmarks for head-pose estimation
- CSV-based logging for session records

## Model Comparison

| Task | Method | Why it was chosen |
|---|---|---|
| Face detection | OpenCV Haar cascade | Fast, lightweight, easy to run locally |
| Mobile phone detection | YOLOv8 | Good general-purpose object detection |
| Head pose | MediaPipe FaceMesh | Practical landmark-based head-pose proxy |
| Logging | CSV file | Simple, transparent, easy to review |

## Challenges Faced

- Streamlit render stability during live refresh
- Balancing detection speed with a smooth monitoring window
- Optional dependency issues for heavier face-recognition stacks
- Keeping the dashboard portable on Windows

## Future Improvements

- Add true face verification with a dedicated identity model
- Switch to `streamlit-webrtc` for smoother real-time streaming
- Add chart-based analytics for session trends
- Export reports as PDF or HTML
- Add anti-spoofing / liveness detection

## Demo Assets

For GitHub submission, add:

- Screenshots under `assets/screenshots/`
- A short demo video under `assets/demo/`

Suggested files:

- `assets/screenshots/dashboard-home.png`
- `assets/screenshots/live-alerts.png`
- `assets/screenshots/session-summary.png`
- `assets/demo/fraud-detection-demo.mp4`

## Final Presentation Outline

For a 5-minute presentation:

1. Problem statement and use case
2. System architecture
3. Detection modules and scoring logic
4. Live demo of webcam or recorded video
5. Logs, summary, limitations, and future work

## Submission Checklist

- Clean repository structure
- `README.md` with setup steps and methodology
- Demo screenshots or video
- Clear module separation
- No generated files committed to GitHub

## Notes

- The app works with webcam or uploaded video.
- Fraud logs are stored in `logs/fraud_log.csv`.
- If `mediapipe` is unavailable, head-pose detection falls back safely.
- If `ultralytics` or the YOLO weights are unavailable, phone detection falls back safely.
