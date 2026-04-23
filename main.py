print("🔥 FILE IS EXECUTING 🔥")
print("importing cv2...")
import cv2
print("importing numpy...")
import numpy as np
print("importing mediapipe...")
import mediapipe as mp
print("importing sounddevice...")
import sounddevice as sd
print("importing threading...")
import threading
print("importing time...")
import time
print("importing json...")
import json

print("importing detector...")
from detector import get_detections
print("importing calibration...")
from calibration import load_calibration
print("importing emotion...")
from emotion import load_emotion_model, detect_emotion
print("importing scorer...")
from scorer import compute_alert, ALERT_SOFT, ALERT_CRITICAL, ALERT_MULTIPLE, ALERT_NONE
print("importing logger...")
from logger import init_db, log_event

print("🔥IMPORTS SUCCESSFULLY🔥")
STATE_FILE = "drivesense_state.json"

# ── ALERT SOUNDS ──────────────────────────────────────────────────────────────

def _beep(frequency=440, duration=0.3):
    t = np.linspace(0, duration, int(44100 * duration))
    wave = 0.3 * np.sin(2 * np.pi * frequency * t)
    sd.play(wave, 44100)
    sd.wait()

def soft_alert():
    threading.Thread(target=_beep, args=(440, 0.4), daemon=True).start()

def critical_alert():
    def _triple():
        for _ in range(3):
            _beep(880, 0.3)
            time.sleep(0.1)
    threading.Thread(target=_triple, daemon=True).start()


# ── DASHBOARD STATE WRITER ────────────────────────────────────────────────────

def write_state(detections, emotion, result):
    """Write current frame state to JSON so dashboard.py can read it live."""
    state = {
        "alert_level": result["level"],
        "ear":         round(detections["ear"],   3) if detections else 0.0,
        "mar":         round(detections["mar"],   3) if detections else 0.0,
        "yaw_dev":     result.get("yaw_dev",   0.0),
        "pitch_dev":   result.get("pitch_dev", 0.0),
        "drowsy":      result["drowsy"],
        "yawning":     result["yawning"],
        "head":        result["head"],
        "emotion":     emotion,
        "timestamp":   time.strftime("%H:%M:%S"),
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"[state] Write failed: {e}")


# ── HUD OVERLAY ───────────────────────────────────────────────────────────────

def draw_hud(frame, detections, emotion, result):
    h, w = frame.shape[:2]
    alert   = result["level"]
    reasons = result["details"]

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

    # Alert colour
    color = {
        ALERT_CRITICAL: (0, 0, 255),
        ALERT_MULTIPLE: (0, 0, 180),
        ALERT_SOFT:     (0, 165, 255),
        ALERT_NONE:     (0, 200, 0),
    }.get(alert, (0, 200, 0))

    cv2.putText(frame, f"Alert: {alert}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Emotion: {emotion}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    if detections:
        cv2.putText(frame, f"EAR:{detections['ear']:.2f}  MAR:{detections['mar']:.2f}",
                    (w - 270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame,
                    f"YawDev:{result['yaw_dev']:.1f}  PitchDev:{result['pitch_dev']:.1f}",
                    (w - 270, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Reason tags at the bottom
    tag_x = 10
    for reason in reasons.split(", "):
        if not reason or reason == "all clear":
            continue
        label = reason.upper().replace("_", " ")
        box_w = len(label) * 10 + 14
        cv2.rectangle(frame, (tag_x, h - 40), (tag_x + box_w, h - 10), (0, 0, 160), -1)
        cv2.putText(frame, label, (tag_x + 5, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tag_x += box_w + 8

    # Critical / Multiple full-frame banner
    if alert in (ALERT_CRITICAL, ALERT_MULTIPLE):
        cv2.rectangle(frame, (0, h // 2 - 40), (w, h // 2 + 40), (0, 0, 180), -1)
        text = "⚠  CRITICAL ALERT" if alert == ALERT_CRITICAL else "⚠  MULTIPLE ALERTS"
        cv2.putText(frame, text, (w // 2 - 160, h // 2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

    return frame


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    print("[DriveSense] Initializing...")
    init_db()

    # ── Calibration ──────────────────────────────────────────────────────────
    baseline = load_calibration()
    if baseline is None:
        print("[DriveSense] No calibration found — run calibration.py first.")
        return

    # ── Load models ───────────────────────────────────────────────────────────
    print("[DriveSense] Loading emotion model...")
    emotion_model = load_emotion_model()

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[DriveSense] Monitoring started. Press Q to quit.")

    ALERT_COOLDOWN  = 5     # seconds between audio alerts
    EMOTION_EVERY   = 30    # run emotion model every N frames
    LOG_EVERY       = 10    # log to SQLite every N frames

    last_alert_time = 0
    frame_count     = 0
    current_emotion = "neutral"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── Landmark detection ────────────────────────────────────────────────
        detections = get_detections(frame, face_mesh)

        # ── Emotion (every EMOTION_EVERY frames) ─────────────────────────────
        if frame_count % EMOTION_EVERY == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) > 0:
                current_emotion, _ = detect_emotion(frame, faces[0], emotion_model)

        # ── Scoring ───────────────────────────────────────────────────────────
        if detections:
            result = compute_alert(
                ear     = detections["ear"],
                mar     = detections["mar"],
                yaw     = detections["yaw"],
                pitch   = detections["pitch"],
                emotion = current_emotion,
            )
        else:
            # No face visible — treat as unknown, don't fire alerts
            result = {
                "level":     ALERT_NONE,
                "drowsy":    False,
                "yawning":   False,
                "head":      False,
                "emotion":   False,
                "details":   "no face",
                "yaw_dev":   0.0,
                "pitch_dev": 0.0,
            }

        # ── Write dashboard state ─────────────────────────────────────────────
        write_state(detections, current_emotion, result)

        # ── Audio alerts (with cooldown) ──────────────────────────────────────
        now = time.time()
        if now - last_alert_time > ALERT_COOLDOWN:
            if result["level"] == ALERT_MULTIPLE:
                critical_alert()
                soft_alert()
                last_alert_time = now
            elif result["level"] == ALERT_CRITICAL:
                critical_alert()
                last_alert_time = now
            elif result["level"] == ALERT_SOFT:
                soft_alert()
                last_alert_time = now

        # ── SQLite logging ────────────────────────────────────────────────────
        if frame_count % LOG_EVERY == 0 and detections:
            log_event(
                ear         = detections["ear"],
                mar         = detections["mar"],
                pitch       = detections["pitch"],
                yaw         = detections["yaw"],
                phone       = False,
                emotion     = current_emotion,
                risk_score  = 0,            # scorer no longer uses numeric score
                alert_level = result["level"],
            )

        # ── Draw HUD and display ──────────────────────────────────────────────
        frame = draw_hud(frame, detections, current_emotion, result)
        cv2.imshow("DriveSense — Driver Monitor", frame)

        frame_count += 1
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[DriveSense] Session ended.")

if __name__ == "__main__":
    print("➡ Calling main()")
    main()
