import os
import numpy as np
import tensorflow as tf
from tensorflow import keras




fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels)=fashion_mnist.load_data()

def reshape_and_normalize(images):
    images = images.reshape(60000, 28, 28, 1)
    images = images / 255.0

    return images


training_images = reshape_and_normalize(training_images)


print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:

            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True

        elif epoch >= 10:
            self.model.stop_training = True


        else:
            pass


def convolutional_model():
    model = tf.keras.models.Sequential([

        #Model
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')

    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

model = convolutional_model()


callbacks = myCallback()

history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
model.evaluate(test_images)