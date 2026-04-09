import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPool2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ⚠️ Dummy training (replace later with real dataset)
import numpy as np
x_dummy = np.random.rand(10,224,224,3)
y_dummy = np.random.randint(0,2,10)

model.fit(x_dummy, y_dummy, epochs=1)

model.save('model.h5')

print("Model saved!")