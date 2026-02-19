import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ---------------- CONFIG ----------------
DATA_DIR = "data/train/smn"
MODEL_DIR = "models/smn"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 15
LR = 0.0001

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- DATA ----------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("CLASS INDICES:", train_gen.class_indices)

# ---------------- MODEL ----------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer=Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ---------------- SAVE ----------------
model_json = model.to_json()
with open(f"{MODEL_DIR}/model.json", "w") as f:
    f.write(model_json)

model.save_weights(f"{MODEL_DIR}/model.h5")

print("✅ SMN model saved successfully")
