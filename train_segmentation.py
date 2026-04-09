import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 256

# =========================
# 🧠 MODEL (DeepLabV3+ FIXED)
# =========================
def DeeplabV3():
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False  # freeze backbone

    x = base_model.output  # (None, 8, 8, 2048)

    # ================= ASPP =================
    x1 = layers.Conv2D(256, 1, padding='same', activation='relu')(x)
    x2 = layers.Conv2D(256, 3, dilation_rate=6, padding='same', activation='relu')(x)
    x3 = layers.Conv2D(256, 3, dilation_rate=12, padding='same', activation='relu')(x)
    x4 = layers.Conv2D(256, 3, dilation_rate=18, padding='same', activation='relu')(x)

    x = layers.Concatenate()([x1, x2, x3, x4])

    # ================= UPSAMPLING (FIXED TO 256x256) =================
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)   # 8 → 32
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)   # 32 → 64
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)   # 64 → 128
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)   # 128 → 256

    output = layers.Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    print("✅ Output shape:", model.output_shape)

    return model


# =========================
# 📁 LOAD DATA (FIXED)
# =========================
def load_segmentation_data(base_path):
    images = []
    masks = []

    classes = ["glioma", "meningioma", "pituitary"]

    for cls in classes:
        folder = os.path.join(base_path, cls)

        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue

        for file in os.listdir(folder):

            if "_mask" not in file:
                img_path = os.path.join(folder, file)

                # 🔥 Handle both jpg & png masks
                name = file.split('.')[0]
                mask_jpg = os.path.join(folder, name + "_mask.jpg")
                mask_png = os.path.join(folder, name + "_mask.png")

                mask_path = mask_jpg if os.path.exists(mask_jpg) else mask_png

                if os.path.exists(mask_path):

                    img = cv2.imread(img_path)
                    mask = cv2.imread(mask_path, 0)

                    if img is None or mask is None:
                        continue

                    # Resize
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

                    # 🔥 Binarize mask CORRECTLY
                    mask = (mask > 127).astype("float32")
                    mask = np.expand_dims(mask, axis=-1)

                    images.append(img)
                    masks.append(mask)

    return np.array(images), np.array(masks)


# =========================
# 🚀 LOAD DATA
# =========================
X, y = load_segmentation_data("Brain_Tumor_dataset/Segmentation")

print("📊 Dataset shape:", X.shape, y.shape)

if len(X) == 0:
    raise ValueError("❌ Dataset is empty. Check your path or mask naming.")

# =========================
# 🔀 TRAIN / VALIDATION SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 🎯 LOSS (DICE + BCE 🔥 BEST)
# =========================
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
           (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# =========================
# ⚙️ COMPILE
# =========================
model = DeeplabV3()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=combined_loss,
    metrics=['accuracy', dice_coef]
)

model.summary()

# =========================
# 📈 CALLBACKS
# =========================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "deeplab_best.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]

# =========================
# 🚀 TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8,
    callbacks=callbacks
)

# =========================
# 💾 SAVE FINAL MODEL
# =========================
model.save("deeplab_final.h5")

print("✅ Training Complete")