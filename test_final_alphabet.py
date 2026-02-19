import cv2
import numpy as np
import pyttsx3
import threading
from tensorflow.keras.models import load_model

from sentence.sentence import SentenceBuilder

# =============================
# CONFIG
# =============================
MODEL_PATH = "final_models/alphabets/model.h5"
LABELS_PATH = "final_models/alphabets/labels.txt"
IMG_SIZE = 128
CONF_THRESHOLD = 0.50

print("Controls:")
print("C → Confirm letter")
print("S → Space")
print("V → Speak sentence 🔊")
print("ESC → Exit")

# =============================
# LOAD MODEL
# =============================
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    LABELS = [line.strip() for line in f]

# =============================
# TEXT TO SPEECH (FINAL FIX)
# =============================
def speak_text(text):
    engine = pyttsx3.init()      # NEW engine every time
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not working")
    exit()

# =============================
# SENTENCE BUILDER
# =============================
sentence_builder = SentenceBuilder()
last_prediction = None

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI
    x1, y1, x2, y2 = 220, 20, 520, 320
    roi = frame[y1:y2, x1:x2]

    # =============================
    # PREPROCESS
    # =============================
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    img = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    # =============================
    # PREDICTION
    # =============================
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    confidence = preds[idx]

    if confidence >= CONF_THRESHOLD:
        last_prediction = LABELS[idx]
    else:
        last_prediction = None

    # =============================
    # KEY CONTROLS
    # =============================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and last_prediction:
        sentence_builder.add_letter(last_prediction)
        print("Accepted:", last_prediction)

    elif key == ord('s'):
        sentence_builder.add_space()
        print("Space added")

    elif key == ord('v'):
        sentence = sentence_builder.get_sentence()
        if sentence:
            print("🔊 Speaking:", sentence)
            threading.Thread(
                target=speak_text,
                args=(sentence,),
                daemon=True
            ).start()

    elif key == 27:
        break

    # =============================
    # UI
    # =============================
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Letter: {last_prediction if last_prediction else 'Detecting'}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Word: {sentence_builder.current_word}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Sentence: {sentence_builder.get_sentence()}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("Alphabet → Sentence → Speech", frame)
    cv2.imshow("Processed ROI", thresh)

cap.release()
cv2.destroyAllWindows()
