# DriveSense — AI-Based Real-Time Driver Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-orange)](https://mediapipe.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-red)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-pink)](https://streamlit.io/)

> B.Tech Major Project — 8th Semester  
> Department of CSIT, Sagar Institute of Research & Technology, Bhopal  
> RGPV, Session 2025–26

---

## What is DriveSense?

DriveSense is a real-time, software-based Driver Monitoring System that uses a
standard USB webcam to detect driver drowsiness, yawning, head turning, and
emotional distraction — and alerts the driver before an accident occurs.

---

## Features

- **Eye Aspect Ratio (EAR)** — drowsiness detection
- **Mouth Aspect Ratio (MAR)** — yawn detection  
- **Head Pose via solvePnP** — lateral head turn and nodding detection  
  *(with calibrated baseline — works at any camera mounting angle)*
- **Emotion Recognition** — mini XCEPTION CNN trained on FER2013
- **4-Level Alert System** — NONE / SOFT / CRITICAL / MULTIPLE with audio
- **Streamlit Dashboard** — live monitor + 7-day weekly analytics
- **SQLite Logging** — persistent session event storage
- **15-Second Personalised Calibration** — adapts to each driver

---

## Project Structure
drivesense/
├── main.py            # Main loop — ties everything together
├── detector.py        # MediaPipe face mesh, EAR, MAR, head pose
├── calibration.py     # 15-sec webcam calibration → calibration.json
├── emotion.py         # FER2013 mini XCEPTION emotion inference
├── scorer.py          # Alert fusion logic (NONE/SOFT/CRITICAL/MULTIPLE)
├── logger.py          # SQLite event logging
├── dashboard.py       # Streamlit live monitor + weekly report
├── requirements.txt   # Python dependencies
└── README.md

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/drivesense.git
cd drivesense
```

### 2. Create conda environment
```bash
conda create -n drivesense_final python=3.10
conda activate drivesense_final
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the emotion model
Download `mini_XCEPTION.102-0.66.hdf5` from the FER2013 model releases
and place it in the project root, or update the path in `emotion.py`.

---

## Running the System

### Step 1 — Run calibration (once per camera position)
```bash
python calibration.py
```
Sit normally, look straight at the camera for 15 seconds.

### Step 2 — Start monitoring
```bash
python main.py
```

### Step 3 — Open dashboard (in a separate terminal)
```bash
streamlit run dashboard.py
```

Then open `http://localhost:8501` in your browser.

---

## Alert Levels

| Level    | Trigger                          | Audio              |
|----------|----------------------------------|--------------------|
| NONE     | All clear                        | Silent             |
| SOFT     | Yawning or emotion flag          | Single 440 Hz tone |
| CRITICAL | Drowsy or head turned            | Triple 880 Hz × 3  |
| MULTIPLE | 3 or more flags simultaneously   | Critical + Soft    |

---

## Requirements

See `requirements.txt`. Key packages:
opencv-python
mediapipe==0.10.9
tensorflow==2.13
numpy==1.24.3
streamlit==1.28
plotly
pandas
sounddevice
scipy
streamlit-autorefresh
---

## Results

| Modality         | Detection Rate |
|------------------|---------------|
| EAR (drowsy)     | 93.3%         |
| MAR (yawn)       | 96.0%         |
| Head turn (yaw)  | 96.7%         |
| Head nod (pitch) | 92.0%         |
| Emotion (FER)    | 75.0%         |
| Multiple triggers| 93.3%         |
| **Overall**      | **91.7%**     |

---

## License

This project is submitted as a B.Tech Major Project.  
For academic use only.