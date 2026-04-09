import os
import cv2
import numpy as np

IMG_SIZE = 224
DATASET_PATH = "Brain_Tumor_dataset/segmentation"

images = []
masks = []

for tumor_type in os.listdir(DATASET_PATH):
    tumor_path = os.path.join(DATASET_PATH, tumor_type)

    if not os.path.isdir(tumor_path):
        continue

    for file in os.listdir(tumor_path):
        if "_mask" in file:
            continue

        img_path = os.path.join(tumor_path, file)
        mask_name = file.split(".")[0] + "_mask.png"
        mask_path = os.path.join(tumor_path, mask_name)

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        img = img / 255.0
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)

        images.append(img)
        masks.append(mask)

X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array(masks).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Loaded Data:", X.shape)

from tensorflow.keras import layers, models

def unet_model():
    inputs = layers.Input((224, 224, 1))

    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    # Bottleneck
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)

    # Decoder
    u1 = layers.UpSampling2D()(c2)
    u1 = layers.concatenate([u1, c1])
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c3)

    model = models.Model(inputs, outputs)
    return model


model = unet_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training Started...")

model.fit(X, Y, epochs=10, batch_size=8)

model.save("unet_model.h5")

print("Training Completed & Model Saved!")