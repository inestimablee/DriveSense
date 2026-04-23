import cv2
import json
import numpy as np
import mediapipe as mp
from detector import get_detections

CALIBRATION_FILE = "calibration.json"
CALIBRATION_SECONDS = 15

def calibrate():
    print("[DriveSense] Starting calibration — look at the camera normally...")
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    total_frames = int(CALIBRATION_SECONDS * fps)

    ears, mars, pitches, yaws = [], [], [], []
    frame_count = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        data = get_detections(frame, mp_face_mesh)
        if data:
            ears.append(data["ear"])
            mars.append(data["mar"])
            pitches.append(data["pitch"])
            yaws.append(data["yaw"])

        # Show countdown on screen
        remaining = int((total_frames - frame_count) / fps) + 1
        cv2.putText(frame, f"Calibrating... {remaining}s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("DriveSense Calibration", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_face_mesh.close()

    if not ears:
        print("[ERROR] No face detected during calibration. Try again.")
        return None

    baseline = {
        "ear_threshold": float(np.mean(ears) * 0.75),
        "mar_threshold": float(np.mean(mars) * 1.5),
        "pitch_range":   [float(np.mean(pitches) - 15), float(np.mean(pitches) + 15)],
        "yaw_range":     [float(np.mean(yaws) - 20),    float(np.mean(yaws) + 20)],
    }

    with open(CALIBRATION_FILE, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"[DriveSense] Calibration done! Baseline saved:")
    print(f"  EAR threshold : {baseline['ear_threshold']:.3f}")
    print(f"  MAR threshold : {baseline['mar_threshold']:.3f}")
    print(f"  Pitch range   : {baseline['pitch_range']}")
    print(f"  Yaw range     : {baseline['yaw_range']}")
    return baseline

def load_calibration():
    try:
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    calibrate()