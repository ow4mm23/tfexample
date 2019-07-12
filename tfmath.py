# coding=utf-8
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from sklearn.preprocessing import MinMaxScaler

## ENABLE EAGER EXECUTION
tf.enable_eager_execution()
# gpu growth ontheway
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
tf.Session(config=conf)

noise=tf.random.normal(shape=[1,32])
# upperb=tf.ones(shape=noise.shape)
# re=K.greater(noise,upperb)
# print(re)
print(noise)

scaler=MinMaxScaler()
noi=scaler.fit_transform(noise.numpy())
print(noi)


input1=Input(shape=(1,32))

def modfy_upper_lower(x,bound=(-1,1)):

    lower=tf.ones(shape=(1,1,32))*bound[0]
    upper=tf.ones(shape=(1,1,32))*bound[1]
    bol_upbound=K.greater(x,upper)
    bol_lowbound=K.less(x,lower)

    x=tf.where(bol_lowbound,lower,x)
    x=tf.where(bol_upbound,upper,x)

    # for i in range(bol_upbound.shape[2]):
    #     if (bol_upbound[0,0,i]):
    #         new_left=tf.slice(x,[0,0,0],[1,1,i])
    #         new_right=tf.slice(x,[0,0,i+1],[1,1,bol_upbound.shape[2]-(i+1)])
    #         x=tf.concat([new_left,tf.slice(upper,[0,0,i],[1,1,1]),new_right],2)
            
    #     if (bol_lowbound[0,0,i]):
    #         new_left=tf.slice(x,[0,0,0],[1,1,i])
    #         new_right=tf.slice(x,[0,0,i+1],[1,1,bol_upbound.shape[2]-(i+1)])
    #         x=tf.concat([new_left,tf.slice(lower,[0,0,i],[1,1,1]),new_right],2)
            
    return x

noise=modfy_upper_lower(noise,bound=(0,1))
print(noise)

print('END')
