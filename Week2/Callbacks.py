#Learning about callbacks -> Stop training when we reach certain point.
#In every epoch, we callback to a code function, check the  metrics, and continue or cancel the training.

import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss')<0.4):
            print("º\nLoss is low, cancel training!")
            self.model.stop_training = True


callbacks = myCallback()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(training_images,training_labels,epochs=5,callbacks=[callbacks])


