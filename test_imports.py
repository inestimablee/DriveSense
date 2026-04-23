import sys
print("1. cv2...")
import cv2
print("2. numpy...")
import numpy as np
print("3. mediapipe...")
import mediapipe as mp
print("4. sounddevice...")
import sounddevice as sd
print("5. detector...")
from detector import get_detections
print("6. calibration...")
from calibration import load_calibration
print("7. emotion...")
from emotion import load_emotion_model, detect_emotion
print("8. scorer...")
from scorer import compute_alert, ALERT_SOFT, ALERT_CRITICAL, ALERT_MULTIPLE, ALERT_NONE
print("9. logger...")
from logger import init_db, log_event
print("✅ ALL IMPORTS OK")