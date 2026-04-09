from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
import cv2
import numpy as np
import tensorflow as tf
from flask_sqlalchemy import SQLAlchemy
from PIL import Image

db = SQLAlchemy()

# ================= DATABASE ================= #

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))
    centre_name = db.Column(db.String(200))
    address = db.Column(db.String(300))
    contact_no = db.Column(db.String(20))
    radiologist_name = db.Column(db.String(200))
    license_no = db.Column(db.String(100))
    logo = db.Column(db.String(200))
    signature = db.Column(db.String(200))
    role = db.Column(db.String(20), default="doctor")
    email = db.Column(db.String(120))
    website = db.Column(db.String(120))
    subscription_start = db.Column(db.DateTime, default=datetime.utcnow)
    subscription_end = db.Column(db.DateTime)
    subscription_plan = db.Column(db.String(50))  # 30d, 3m, 6m, 12m


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    result = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    patient_name = db.Column(db.String(100))
    sex = db.Column(db.String(10))
    age = db.Column(db.String(10))
    mobile = db.Column(db.String(20))
    ref_by = db.Column(db.String(100))
    image = db.Column(db.String(200))
    segmented_image = db.Column(db.String(200))
    date = db.Column(db.String(50))
    patient_id = db.Column(db.String(50))
    tumor_area = db.Column(db.Float)

# ================= CUSTOM LOSSES (DEFINE FIRST) ================= #

def focal_loss(y_true, y_pred):
    alpha = 0.8
    gamma = 2.0
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = tf.exp(-bce)
    return alpha * (1 - bce_exp) ** gamma * bce

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
           (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

# ================= LOAD MODELS ================= #

cnn_model = None
deeplab_model = None

def load_models():
    global cnn_model, deeplab_model

    if cnn_model is None:
        cnn_model = load_model("best_model.h5", compile=False)

    if deeplab_model is None:
        deeplab_model = load_model("deeplab_final.keras", compile=False)
    print("DEEPLAB MODEL:", type(deeplab_model))
# ================= PREPROCESS ================= #

def preprocess_cnn(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ================= CLASSIFICATION ================= #

def detect_tumor(img):
    load_models()
    pred = cnn_model.predict(img, verbose=0)[0]

    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    class_index = np.argmax(pred)

    class_name = class_names[class_index]
    confidence = float(np.max(pred))

    return (class_name != "notumor"), class_name, confidence

# ================= SEGMENTATION ================= #

def segment_tumor(image_path):
    load_models()
    original = cv2.imread(image_path)

    if original is None:
        raise Exception("Image not loaded properly")

    h, w = original.shape[:2]

    # --- preprocess ---
    img = cv2.resize(original, (256, 256))
    img = img / 255.0
    img_input = np.expand_dims(img, axis=0)

    # --- predict ---
    pred = deeplab_model.predict(img_input, verbose=0)[0]

    # 🔥 IMPORTANT FIX
    mask = (pred > 0.5).astype("uint8").squeeze()

    # resize back
    mask = cv2.resize(mask, (w, h))

    # clean noise
    mask = cv2.medianBlur(mask, 5)

    # area
    tumor_area = int(np.sum(mask > 0))

    # overlay
    overlay = original.copy()
    overlay[mask > 0] = [0, 0, 255]

    output = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)

    # bounding box
    contours, _ = cv2.findContours(
        mask.astype("uint8"),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

    # save
    base, _ = os.path.splitext(image_path)
    output_path = base + "_segmented.png"
    cv2.imwrite(output_path, output)

    return output_path, tumor_area