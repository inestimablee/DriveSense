import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import urllib.request
import os

MODEL_PATH = "emotion_model.h5"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Risk multipliers — angry/fear/sad drivers are higher risk
EMOTION_RISK = {
    "Angry":    1.4,
    "Disgust":  1.1,
    "Fear":     1.3,
    "Happy":    0.8,
    "Sad":      1.2,
    "Surprise": 1.1,
    "Neutral":  1.0,
}

def download_model():
    url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
    print("[DriveSense] Downloading emotion model...")
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("[DriveSense] Emotion model downloaded!")

def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        download_model()
    return load_model(MODEL_PATH, compile=False)

def detect_emotion(frame, face_coords, model):
    x, y, w, h = face_coords
    face = frame[y:y+h, x:x+w]
    if face.size == 0:
        return "Neutral", 1.0

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (64, 64))
    face = face.astype("float32") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    preds = model.predict(face, verbose=0)[0]
    emotion = EMOTIONS[np.argmax(preds)]
    multiplier = EMOTION_RISK[emotion]
    return emotion, multiplier