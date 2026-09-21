import tensorflow as tf
import numpy as np
import os

# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = "models/cat_dog_cnn.keras"

IMAGE_PATH = "test_images/bili.jpg"

IMAGE_SIZE = (128, 128)


# ============================================================
# 2. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model file not found!")

    print(
        f"Expected model location: {MODEL_PATH}"
    )

    print(
        "\nPlease run train.py first."
    )

    exit()


# ============================================================
# 3. CHECK TEST IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):

    print("ERROR: Test image not found!")

    print(
        f"Expected image location: {IMAGE_PATH}"
    )

    print(
        "\nPlease put your image inside the test_images folder."
    )

    exit()


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

print("Loading CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# 5. LOAD IMAGE
# ============================================================

print("\nLoading image...")

image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMAGE_SIZE
)


# ============================================================
# 6. CONVERT IMAGE TO NUMPY ARRAY
# ============================================================

image_array = tf.keras.utils.img_to_array(
    image
)


# ============================================================
# 7. ADD BATCH DIMENSION
# ============================================================

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# 8. MAKE PREDICTION
# ============================================================

prediction = model.predict(
    image_array,
    verbose=0
)[0][0]


# ============================================================
# 9. DISPLAY RAW PREDICTION
# ============================================================

print("\n======================================")
print("RAW PREDICTION")
print("======================================")

print(
    f"Raw prediction value: {prediction:.6f}"
)


# ============================================================
# 10. CONVERT PREDICTION TO CLASS
# ============================================================

# IMPORTANT:
#
# 0 = CAT
# 1 = DOG
#
# Therefore:
#
# prediction < 0.5  --> CAT
# prediction >= 0.5 --> DOG


if prediction >= 0.5:

    predicted_class = "DOG"

    dog_probability = prediction

    cat_probability = 1 - prediction

else:

    predicted_class = "CAT"

    cat_probability = 1 - prediction

    dog_probability = prediction


# ============================================================
# 11. DISPLAY RESULT
# ============================================================

print("\n======================================")
print("FINAL PREDICTION")
print("======================================")

print(
    f"Prediction: {predicted_class}"
)

print(
    f"DOG Probability: {dog_probability:.2%}"
)

print(
    f"CAT Probability: {cat_probability:.2%}"
)

print(
    f"\nRaw Probability: {prediction:.6f}"
)

print("======================================")