import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
from sentence.sentence import SentenceBuilder
import os

# =============================
# CONFIG
# =============================
MODEL_PATH = "final_models/alphabets/model.h5"
LABELS_PATH = "final_models/alphabets/labels.txt"
IMG_SIZE = 128
CONF_THRESHOLD = 0.50

# UI sizes
WINDOW_W = 1200
WINDOW_H = 700

LEFT_W = 800
RIGHT_W = 400

CAM_H = 450

IMAGE_PATH = "signs.png"   # image for right side

# =============================
# LOAD MODEL & LABELS
# =============================
model = load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    LABELS = [l.strip() for l in f]

# =============================
# LOAD IMAGE (RIGHT PANEL)
# =============================
ui_image = None
if os.path.exists(IMAGE_PATH):
    ui_image = cv2.imread(IMAGE_PATH)
    ui_image = cv2.resize(ui_image, (RIGHT_W, WINDOW_H))

# =============================
# TTS
# =============================
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak(text, save=False):
    if not text.strip():
        return
    if save:
        engine.save_to_file(text, "output_speech.wav")
    engine.say(text)
    engine.runAndWait()

# =============================
# SENTENCE BUILDER
# =============================
sentence_builder = SentenceBuilder()

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not working")
    exit()

print("✅ App started")

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (LEFT_W, CAM_H))

    # ROI (adjusted for new size)
    x1, y1, x2, y2 = 260, 40, 540, 320
    roi = frame[y1:y2, x1:x2]

    # PREPROCESS
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    img = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    conf = preds[idx]

    current_letter = LABELS[idx] if conf >= CONF_THRESHOLD else ""

    # =============================
    # KEY INPUT
    # =============================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and current_letter:
        sentence_builder.add_letter(current_letter)

    elif key == ord('s'):
        sentence_builder.add_space()

    elif key == ord('x'):
        sentence_builder.clear_word()

    elif key == ord('z'):
        sentence_builder.clear_sentence()

    elif key == ord('v'):
        speak(sentence_builder.get_sentence())

    elif key == ord('d'):
        speak(sentence_builder.get_sentence(), save=True)

    elif key == 27:
        break

    # =============================
    # MAIN UI CANVAS
    # =============================
    ui = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    ui[:] = (30, 30, 30)

    # LEFT: CAMERA
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ui[0:CAM_H, 0:LEFT_W] = frame

    # LEFT: TEXT
    cv2.putText(ui, f"Detected Alphabet: {current_letter}",
                (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.putText(ui, f"Current Word: {sentence_builder.current_word}",
                (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    cv2.putText(ui, f"Sentence: {sentence_builder.get_sentence()}",
                (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(ui,
                "C=Confirm  S=Space  V=Speak  D=Download  X=ClearWord  Z=ClearSentence  ESC=Exit",
                (20, 660),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # RIGHT: IMAGE
    if ui_image is not None:
        ui[:, LEFT_W:WINDOW_W] = ui_image

    cv2.imshow("Sign Language to Text & Speech", ui)

# =============================
# CLEANUP
# =============================
cap.release()
cv2.destroyAllWindows()
engine.stop()
