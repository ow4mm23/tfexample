# -*-coding:utf-8-*-
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import sys
import tensorflow as tf
import numpy as np

# DISABLE GPU DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "{}"

# gpu growth ontheway
# tf.verison == 1.14
# conf = tf.ConfigProto()
# conf.gpu_options.allow_growth = True
# tf.Session(config=conf)
# tf.verison == 1.14 END
# tf.version == 2.0
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
# tf.version == 2.0

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

eps = 50

# with tf.device('/CPU:0'):
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

tbcallback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    # write_grads=True,
    write_graph=True)
history = model.fit(x_train, y_train, epochs=eps, callbacks=[])
# tf.nn.dropout()
# model.summary.histogram()
model.evaluate(x_test, y_test, verbose=2)
# model.save('xx.h5')

# visual graph
# writer=tf.summary.FileWriter('./logs',tf.get_default_graph())
# writer.close()
