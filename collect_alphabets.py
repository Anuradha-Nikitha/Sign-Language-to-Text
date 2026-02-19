import cv2
import os

BASE_DIR = "data/train/alphabets"
LETTERS = list("A B C E F G H J L O P Q V W X Y Z".split())

IMG_SIZE = 128
MAX_IMAGES = 300

# Create folders
for l in LETTERS:
    os.makedirs(os.path.join(BASE_DIR, l), exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("Press letter key to capture image")
print("ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI
    x1, y1, x2, y2 = 220, 20, 520, 320
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 2)
    _, thresh = cv2.threshold(
        blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    thresh = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))

    cv2.imshow("Frame", frame)
    cv2.imshow("Processed", thresh)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    char = chr(key).upper()
    if char in LETTERS:
        path = os.path.join(BASE_DIR, char)
        count = len(os.listdir(path))

        if count < MAX_IMAGES:
            cv2.imwrite(f"{path}/{count}.jpg", thresh)
            print(f"Saved {char}: {count+1}")
        else:
            print(f"{char} completed")

cap.release()
cv2.destroyAllWindows()
