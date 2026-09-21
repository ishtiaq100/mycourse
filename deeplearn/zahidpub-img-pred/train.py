import os
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# 1. SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cat_dog_cnn.keras")


# ============================================================
# 2. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)


# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 5. CHECK CLASS NAMES
# ============================================================

print("\n======================================")
print("CLASS NAMES")
print("======================================")

print(train_dataset.class_names)

print("\nClass mapping:")

for index, class_name in enumerate(train_dataset.class_names):
    print(index, "=", class_name)


# ============================================================
# 6. PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.1
    ),

    tf.keras.layers.RandomZoom(
        0.1
    )
])


# ============================================================
# 8. CREATE CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    # Input
    tf.keras.layers.Input(
        shape=(128, 128, 3)
    ),

    # Data augmentation
    data_augmentation,

    # Normalize pixel values
    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    # --------------------------------------------------------
    # CNN Layer 1
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # CNN Layer 2
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # CNN Layer 3
    # --------------------------------------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Convert feature maps to one-dimensional vector
    # --------------------------------------------------------

    tf.keras.layers.Flatten(),

    # --------------------------------------------------------
    # Fully Connected Layer
    # --------------------------------------------------------

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Prevent overfitting
    tf.keras.layers.Dropout(
        0.5
    ),

    # --------------------------------------------------------
    # Output Layer
    # 0 = Cat
    # 1 = Dog
    # --------------------------------------------------------

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 9. DISPLAY MODEL
# ============================================================

print("\n======================================")
print("CNN MODEL")
print("======================================")

model.summary()


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n======================================")
print("STARTING TRAINING")
print("======================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n======================================")
print("MODEL SAVED")
print("======================================")

print(f"Model saved at: {MODEL_PATH}")


# ============================================================
# 13. EVALUATE MODEL
# ============================================================

print("\n======================================")
print("MODEL EVALUATION")
print("======================================")

validation_loss, validation_accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print(
    f"\nValidation Loss: {validation_loss:.4f}"
)

print(
    f"Validation Accuracy: {validation_accuracy:.4f}"
)

print(
    f"Validation Accuracy: {validation_accuracy * 100:.2f}%"
)


# ============================================================
# 14. PLOT ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 15. PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.show()


print("\n======================================")
print("TRAINING COMPLETED")
print("======================================")