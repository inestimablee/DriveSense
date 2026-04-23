import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe landmark indices for eyes and mouth
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291, 39,  181, 0,   17,  269, 405]

# Head pose key points
NOSE_TIP    = 1
CHIN        = 152
LEFT_EYE_L  = 263
RIGHT_EYE_R = 33
LEFT_MOUTH  = 61
RIGHT_MOUTH = 291

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    A = distance.euclidean(pts[2], pts[6])
    B = distance.euclidean(pts[3], pts[7])
    C = distance.euclidean(pts[0], pts[1])
    return (A + B) / (2.0 * C)

def head_pose(landmarks, w, h):
    # 3D model points of key facial landmarks
    model_points = np.array([
        (0.0,    0.0,    0.0),     # Nose tip
        (0.0,   -330.0, -65.0),    # Chin
        (-225.0, 170.0, -135.0),   # Left eye corner
        (225.0,  170.0, -135.0),   # Right eye corner
        (-150.0,-150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0),   # Right mouth corner
    ], dtype=np.float64)

    # Corresponding 2D image points from landmarks
    lm_indices = [1, 152, 263, 33, 61, 291]
    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in lm_indices
    ], dtype=np.float64)

    # Camera internals
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1        ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        model_points, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat([rotation_mat, np.zeros((3,1))])
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch = float(euler_angles[0])
    yaw   = float(euler_angles[1])

    return pitch, yaw

def get_detections(frame, face_mesh):
    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    ear   = (eye_aspect_ratio(lm, LEFT_EYE, w, h) +
             eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2.0
    mar   = mouth_aspect_ratio(lm, MOUTH, w, h)
    pitch, yaw = head_pose(lm, w, h)

    return {"ear": ear, "mar": mar, "pitch": pitch, "yaw": yaw}