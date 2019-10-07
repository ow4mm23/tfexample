# coding:utf-8
from __future__ import absolute_import,division,print_function,unicode_literals
import os
import tensorflow as tf
# from tensorflow.python.client import device_lib

print("# of GPU Available: ",len(tf.config.experimental.list_physical_devices('GPU')))
