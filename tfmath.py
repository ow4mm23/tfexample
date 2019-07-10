# coding=utf-8
import tensorflow as tf

import tensorflow.keras.backend as K

## ENABLE EAGER EXECUTION
tf.enable_eager_execution()


noise=tf.random.normal(shape=[1,1,32])
# upperb=tf.ones(shape=noise.shape)
# re=K.greater(noise,upperb)
# print(re)
print(noise)

def modfy_upper_lower(x,bound=(-1,1)):

    lower=tf.ones(shape=(1,1,32))*bound[0]
    upper=tf.ones(shape=(1,1,32))*bound[1]
    bol_upbound=K.greater(x,upper)
    bol_lowbound=K.less(x,lower)
    
    for i in range(bol_upbound.shape[2]):
        if (bol_upbound.numpy()[0,0,i]==True):
            new_left=tf.slice(x,[0,0,0],[1,1,i])
            new_right=tf.slice(x,[0,0,i+1],[1,1,bol_upbound.shape[2]-(i+1)])
            x=tf.concat([new_left,tf.slice(upper,[0,0,i],[1,1,1]),new_right],2)
            
        if (bol_lowbound.numpy()[0,0,i]==True):
            new_left=tf.slice(x,[0,0,0],[1,1,i])
            new_right=tf.slice(x,[0,0,i+1],[1,1,bol_upbound.shape[2]-(i+1)])
            x=tf.concat([new_left,tf.slice(lower,[0,0,i],[1,1,1]),new_right],2)
            
    return x

noise=modfy_upper_lower(noise,bound=(0,1))
print(noise)

print('END')
