#CONVOLUTIONS -> Pixel + immediate neighbours passed by a filter.
#We combine convolutions with Pooling:
#POOLING -> Compressing a image.
#pe: 4x4 pixel image -> make 4 2x2 pixel images -> take the bigger value of those 4 to make 1 2x2.
#so we went from 4x4 to 2x2
#We filter and compress so we don't have to study pixel by pixel the images, now we try to identify things from the image to categorize it.

import tensorflow as tf
from tensorflow import keras

#IMPLEMENTING CONVOLUTIONAL LAYERS.

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',input_shape=(28,28,1)),
    #LAYEER CONV2D. Creates 64 filters of 3x3 pixels, activated by relu (only positive values), input shape still is 28x28, 1 means color depth.
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.summary() #inspect the layers and see the journey of the data.


#https://bit.ly/2UGa7uH
