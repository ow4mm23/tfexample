# coding=utf-8
import os
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model

from sklearn.preprocessing import MinMaxScaler

## ENABLE EAGER EXECUTION
tf.enable_eager_execution()
# DISABLE GPU DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "{}"
# gpu growth ontheway
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
tf.Session(config=conf)

def build_gan_model():
    main_input=Input(shape=(64,),name='main_input')
    x=Dense(64,activation='sigmoid',name='hidden_1')(main_input)

    second_input=Input(shape=(32,),name='second_input')
    x2=Dense(32,activation='softmax',name='hidden_2')(second_input)
    # x2=tf.cast(tf.argmax(x2,2),dtype=tf.float32)
    x2=tf.argmax(x2)

    y=tf.keras.layers.concatenate([x,x2])

    return Model(inputs=[main_input,second_input],outputs=y)

gan_model=build_gan_model()
gan_model.summary()


print('END')
