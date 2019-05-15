# coding:utf-8

'''
效果不好
'''

import os

from tensorflow import keras
import tensorflow as tf
import tensorboard

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# tf.enable_eager_execution()

# read csvfile
raw_dataset = pd.read_csv(
    './dataset/EMHIRESPV_TSh_CF_Country_19862015.csv', usecols=[0], skiprows=[0])
solar_country = raw_dataset.copy()

train_dataset = solar_country[0:210000]
test_dataset = solar_country.drop(train_dataset.index)


# EUsolardatacsv = ['./dataset/EMHIRESPV_TSh_CF_Country_19862015.csv']
# record_defaults = [tf.float64]
# dataset = tf.data.experimental.make_csv_dataset(
#     EUsolardatacsv,100 )

# plot figure to show
plt.subplot(121)
plt.plot(train_dataset, '-')
# plt.show()

train_dataset = train_dataset.values
test_dataset = test_dataset.values

train_dataset_w = train_dataset.reshape(train_dataset.shape[0], 1, 1)
test_dataset_w = test_dataset.reshape(test_dataset.shape[0], 1, 1)

# create a sequential model
model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=[None, train_dataset.shape[1]]),
    # tf.contrib.seq2seq.BahdanauAttention(num_units=64,memory=r),
    keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dense(64, activation='sigmoid'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(train_dataset_w, epochs=1, verbose=1, callbacks=[
          keras.callbacks.TensorBoard(log_dir='./vslogs', write_grads=True)])

test_pre = model.predict(test_dataset_w)

plt.subplot(122)
plt.plot(test_pre)
# plt.legend()
plt.show()

print('THIS IS END')
