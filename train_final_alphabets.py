import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# ============================
# CONFIG
# ============================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

DATA_DIR = "final_data/train"
MODEL_DIR = "final_models/alphabets"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# DATA GENERATORS
# ============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    # 🔥 RIGHT + LEFT HAND SUPPORT
    horizontal_flip=True,

    # Extra robustness
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print("Classes:", train_gen.class_indices)

# ============================
# MODEL
# ============================
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
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================
# TRAIN
# ============================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ============================
# SAVE MODEL
# ============================
model.save(f"{MODEL_DIR}/model.h5")

with open(f"{MODEL_DIR}/labels.txt", "w") as f:
    for label in train_gen.class_indices:
        f.write(label + "\n")

print("✅ FINAL alphabet model (LEFT + RIGHT HAND) saved successfully")
