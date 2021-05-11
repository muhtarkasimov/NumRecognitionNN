# Setting Seed Random Numbers for Reproducibility
import numpy as np
np.random.seed(123)

# Setting Seed Random Numbers for the TensorFlow Backend
import tensorflow as tf
tf.random.set_seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPool2D, Conv2D
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the images.
train_images = (X_train / 255)
test_images = (X_test / 255)

# Flatten the images.
train_images = X_train.reshape((-1, 784))
test_images = X_test.reshape((-1, 784))


# Build the model
model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(784,)),
    Dense(32, activation='sigmoid'),
    Dense(16, activation='sigmoid'),
    # Dense(16, activation='sigmoid'),
    # Dense(16, activation='sigmoid'),
    Dense(10, activation='softmax'),
])

# Single layer design
# model = Sequential([
#     Dense(10, activation='softmax', input_shape=(784,))
# ])

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model.
model.fit(
    train_images,
    to_categorical(y_train),
    epochs=15,
    batch_size=32,
    validation_split=0.1
)

# Evaluate the model
model.evaluate(
    test_images,
    to_categorical(y_test)
)

# Save the model to disk.
model.save_weights('model.KerasNN')

# Load the model from disk later using:
# model.load_weights('model.KerasNN')

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1))  # [7, 2, 1, 0, 4]

