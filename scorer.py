import json
import numpy as np
from collections import deque
import time

CALIBRATION_FILE = "calibration.json"

# Alert levels
ALERT_NONE     = 0
ALERT_SOFT     = 1
ALERT_CRITICAL = 2
ALERT_MULTIPLE = 3  # multiple things wrong simultaneously

# Risky emotions
RISKY_EMOTIONS = ["Angry", "Fear", "Sad"]

class RiskScorer:
    def __init__(self):
        self.baseline = self.load_baseline()
        self.yawn_times = deque(maxlen=10)   # timestamps of yawns
        self.ear_frames = 0                   # consecutive drowsy frames
        self.YAWN_WINDOW = 60                 # seconds to count frequent yawns
        self.YAWN_CRITICAL = 3               # yawns in window = critical
        self.EAR_FRAMES_SOFT = 5            # consecutive frames for soft alert

    def load_baseline(self):
        try:
            with open(CALIBRATION_FILE) as f:
                return json.load(f)
        except FileNotFoundError:
            print("[WARNING] No calibration found! Using defaults.")
            return {
                "ear_threshold": 0.20,
                "mar_threshold": 0.75,
                "pitch_range":   [-20, 20],
                "yaw_range":     [-25, 25],
            }

    def calculate(self, detections, emotion="Neutral", phone_detected=False):
        if detections is None:
            return 0, ALERT_NONE, {}

        ear   = detections["ear"]
        mar   = detections["mar"]
        pitch = detections["pitch"]
        yaw   = detections["yaw"]

        triggers  = {}
        alerts    = []

        # ── 1. HEAD DOWN/AWAY → immediate critical ──────────────
        p_min, p_max = self.baseline["pitch_range"]
        y_min, y_max = self.baseline["yaw_range"]

        head_down = pitch < (p_min - 5) or pitch > (p_max + 5)
        head_away = yaw < (y_min - 5) or yaw > (y_max + 5)

        if head_down or head_away:
            triggers["head"] = True
            alerts.append(ALERT_CRITICAL)

        # ── 2. FREQUENT YAWNING → critical ───────────────────────
        if mar > self.baseline["mar_threshold"]:
            self.yawn_times.append(time.time())
            triggers["yawning"] = True

        # Count yawns in last 60 seconds
        now = time.time()
        recent_yawns = sum(1 for t in self.yawn_times if now - t < self.YAWN_WINDOW)
        if recent_yawns >= self.YAWN_CRITICAL:
            triggers["frequent_yawn"] = True
            alerts.append(ALERT_CRITICAL)
        elif recent_yawns >= 1:
            alerts.append(ALERT_SOFT)

        # ── 3. EAR BELOW THRESHOLD → soft alert ─────────────────
        if ear < self.baseline["ear_threshold"]:
            self.ear_frames += 1
            triggers["drowsy"] = True
        else:
            self.ear_frames = 0

        if self.ear_frames >= self.EAR_FRAMES_SOFT:
            alerts.append(ALERT_SOFT)

        # ── 4. RISKY EMOTION → soft alert ───────────────────────
        if emotion in RISKY_EMOTIONS:
            triggers["emotion"] = True
            alerts.append(ALERT_SOFT)

        # ── 5. PHONE → critical ──────────────────────────────────
        if phone_detected:
            triggers["phone"] = True
            alerts.append(ALERT_CRITICAL)

        # ── 6. MULTIPLE triggers → special critical ──────────────
        if len(triggers) >= 2:
            final_alert = ALERT_MULTIPLE
        elif ALERT_CRITICAL in alerts:
            final_alert = ALERT_CRITICAL
        elif ALERT_SOFT in alerts:
            final_alert = ALERT_SOFT
        else:
            final_alert = ALERT_NONE

        # Risk score still useful for dashboard
        score = min(len(triggers) * 25 + (20 if final_alert == ALERT_CRITICAL else 0), 100)

        return score, final_alert, triggers
    # Module-level scorer instance
_scorer = RiskScorer()

def compute_alert(ear, mar, yaw, pitch, emotion="Neutral"):
    detections = {"ear": ear, "mar": mar, "yaw": yaw, "pitch": pitch}
    score, level, triggers = _scorer.calculate(detections, emotion)
    return {
        "level":     level,
        "drowsy":    triggers.get("drowsy",   False),
        "yawning":   triggers.get("yawning",  False),
        "head":      triggers.get("head",     False),
        "emotion":   triggers.get("emotion",  False),
        "details":   ", ".join(triggers.keys()) or "all clear",
        "yaw_dev":   yaw,
        "pitch_dev": pitch,
    }