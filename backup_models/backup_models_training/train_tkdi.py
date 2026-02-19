import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os

# ---------------- CONFIG ----------------
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 15

TRAIN_DIR = "data/train/tkdi"
TEST_DIR = "data/test/tkdi"   # can be empty for now
MODEL_DIR = "models/tkdi"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- DATA GENERATORS ----------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.15,
    zoom_range=0.15,
    rotation_range=10,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("CLASS INDICES:", train_data.class_indices)
# Expected order: {'D':0, 'I':1, 'K':2, 'T':3}

# ---------------- MODEL ----------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(4, activation="softmax")  # TKDI = 4 classes
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data if test_data.samples > 0 else None
)

# ---------------- SAVE MODEL ----------------
with open(f"{MODEL_DIR}/model.json", "w") as f:
    f.write(model.to_json())

model.save_weights(f"{MODEL_DIR}/model.h5")

print("✅ TKDI model saved successfully")
