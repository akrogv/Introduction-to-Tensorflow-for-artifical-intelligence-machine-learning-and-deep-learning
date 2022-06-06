import tensorflow as tf
from tensorflow import keras
import numpy as np

#HELLO WORLD! in Tensorflow

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

class Modelo:

    def __init__(self,units=0,input_shape=0,optimizer='sgd',loss='mean_squared_error'):
        self.units=units
        self.input_shape=input_shape
        self.optimizer=optimizer
        self.loss=loss

    def buildmodel (self,units,input_shape,optimizer,loss):
      tf.keras.Sequential([keras.layers.Dense(units=units,input_shape=input_shape)])
      self.compile(optimizer=optimizer,loss=loss)

    def train(self,epochs=0):
      self.fit(xs,ys,epochs=epochs)

    def imprime(self,predict=0):
      print(self.predict([predict]))


if __name__ == '__main__':
    Modelo1 = Modelo()
    Modelo1.buildmodel(1,[1],'sdg','mean_squared_error')




