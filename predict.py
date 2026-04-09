import cv2
import numpy as np
import tensorflow as tf
from models import classifier, segmenter

# Load models
classifier = tf.keras.models.load_model("best_model.h5")
segmenter = tf.keras.models.load_model("deeplab_best.h5", compile=False)

def calculate_tumor_area(mask):
    # mask is binary (0 or 1)
    tumor_pixels = np.sum(mask > 0.5)
    return tumor_pixels

def calculate_real_area(mask, pixel_spacing=0.5):
    # pixel_spacing in mm (MRI metadata usually)
    pixel_area = pixel_spacing * pixel_spacing
    tumor_pixels = np.sum(mask > 0.5)
    return tumor_pixels * pixel_area

def overlay_mask(image, mask):
    mask = (mask > 0.5).astype("uint8")

    # Resize mask to match original image
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Create red mask
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = 255  # Red channel

    # Apply mask
    overlay = cv2.addWeighted(image, 1.0, red_mask * mask[:, :, None], 0.5, 0)

    return overlay


IMG_SIZE = 224
SEG_SIZE = 256

classes = ["glioma", "meningioma", "pituitary", "no_tumor"]

def predict(image_path):
    # ===== Load Image =====
    img = cv2.imread(image_path)

    # ===== CLASSIFICATION =====
    img_cls = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_cls = np.expand_dims(img_cls, axis=0)

    pred = classifier.predict(img_cls)
    class_id = np.argmax(pred)
    class_name = classes[class_id]

    # ===== SEGMENTATION =====
    img_seg = cv2.resize(img, (SEG_SIZE, SEG_SIZE)) / 255.0
    img_seg = np.expand_dims(img_seg, axis=0)

    mask = segmenter.predict(img_seg)[0]

    # ===== AREA =====
    tumor_area = np.sum(mask > 0.5)

    # ===== OVERLAY =====
    overlay = overlay_mask(img, mask)

    # Save output
    output_path = "static/output.png"
    cv2.imwrite(output_path, overlay)

    return class_name, tumor_area, output_path

import cv2
import numpy as np
import tensorflow as tf
import os

# Load segmentation model
segmenter = tf.keras.models.load_model("deeplab_best.h5", compile=False)

SEG_SIZE = 256

def calculate_tumor_area(mask):
    return int(np.sum(mask > 0.5))


def overlay_mask(image, mask):
    mask = (mask > 0.5).astype("uint8")
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    red = np.zeros_like(image)
    red[:, :, 2] = 255

    overlay = cv2.addWeighted(image, 1.0, red * mask[:, :, None], 0.5, 0)
    return overlay


def segment_tumor(image_path):
    img = cv2.imread(image_path)

    # Resize for model
    img_seg = cv2.resize(img, (SEG_SIZE, SEG_SIZE)) / 255.0
    img_seg = np.expand_dims(img_seg, axis=0)

    # Predict mask
    mask = segmenter.predict(img_seg)[0]

    # Calculate area
    tumor_area = calculate_tumor_area(mask)

    # Overlay
    overlay = overlay_mask(img, mask)

    # Save output
    output_path = os.path.join("static", "seg_" + os.path.basename(image_path))
    cv2.imwrite(output_path, overlay)

    return output_path, tumor_area