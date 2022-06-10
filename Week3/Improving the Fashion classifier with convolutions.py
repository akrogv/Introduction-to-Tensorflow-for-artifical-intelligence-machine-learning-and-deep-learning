import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


#load data from keras database.
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels)=fashion_mnist.load_data()

#normalize data
train_images  = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',input_shape=(28,28,1)),
    #LAYEER CONV2D. Creates 64 filters of 3x3 pixels, activated by relu (only positive values), input shape still is 28x28, 1 means color depth.
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.summary()

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)


print(f'\nModel EV:')
test_loss = model.evaluate(test_images,test_labels)