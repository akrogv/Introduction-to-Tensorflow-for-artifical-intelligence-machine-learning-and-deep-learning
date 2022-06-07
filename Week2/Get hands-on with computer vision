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




#check pixel greyscale values from a image label x
index = 0
np.set_printoptions(linewidth=320)
print(f'LABEL: {train_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {train_images[index]}')
plt.imshow(train_images[index],cmap='Greys')



#Model with 3 layers now.
#First one -> Flatten. Makes multidimensional input into one-dimensional.
#each dense layer of neurons has a activation function that tells them what to do.
#RELU MEANS: if x>0 return 0 else 0
#SOFTMAX -> Scales values so the sum of all of those equals 1. WHEN you do that to a model,
#you can think of it as probabilities.


model1 = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model1.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model1.fit(train_images, train_labels, epochs=5)

#model1.evaluate(test_images,test_labels)
