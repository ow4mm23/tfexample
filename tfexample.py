# -*-coding:utf-8-*-
import argparse
import os
import sys
#from __future__ import absolute_import,division,print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# with tf.device('/CPU:0'):
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0, 2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

tbcallback = keras.callbacks.TensorBoard(
    log_dir='./logs', write_grads=True, write_graph=True)
history = model.fit(x_train, y_train, epochs=5, callbacks=[])
# tf.nn.dropout()
# model.summary.histogram()
model.evaluate(x_test, y_test)
# model.save('xx.h5')

# visual graph
# writer=tf.summary.FileWriter('./logs',tf.get_default_graph())
# writer.close()
