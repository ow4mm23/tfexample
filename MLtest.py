# -*- coding:utf-8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import tensorboard

# 读取数据
mnist = input_data.read_data_sets("./mnist/", one_hot=True)
print(mnist.train.num_examples)
print(mnist.validation.num_examples)
print(mnist.test.num_examples)

# # 显示图片
# temp = mnist.train.images[3]
# img = np.reshape(temp, (28, 28))
# plt.imshow(img, cmap='gray')
# plt.show()
# # 显示标签
# ltemp = mnist.train.labels[3]
# print(ltemp)

inputSize = 784
outputSize = 10
hiddenSize = 50
batchSize = 64
trainCycle = 5000

# inputlayer
inputLayer = tf.placeholder(tf.float32, shape=[None, inputSize])

# hiddenlayer
hiddenWeight = tf.Variable(tf.truncated_normal(
    [inputSize, hiddenSize], mean=0, stddev=0.1))
hiddenBias = tf.Variable(tf.truncated_normal([hiddenSize]))
hiddenLayer = tf.add(tf.matmul(inputLayer, hiddenWeight), hiddenBias)
hiddenLayer = tf.nn.sigmoid(hiddenLayer)

# outputlayer
outputWeight = tf.Variable(tf.truncated_normal(
    [hiddenSize, outputSize], mean=0, stddev=0.1))
outputBias = tf.Variable(tf.truncated_normal([outputSize], mean=0, stddev=0.1))
outputLayer = tf.add(tf.matmul(hiddenLayer, outputWeight), outputBias)
outputLayer = tf.nn.sigmoid(outputLayer)

# labels
outputLabel = tf.placeholder(tf.float32, shape=[None, outputSize])

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=outputLabel, logits=outputLayer))

# optimizer
optimizer = tf.train.AdamOptimizer()

# train target
target = optimizer.minimize(loss)

# train start
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(trainCycle):
        batch = mnist.train.next_batch(batchSize)
        sess.run(target, feed_dict={
                 inputLayer: batch[0], outputLabel: batch[1]})

        if i % 1000 == 0:
            corrected = tf.equal(tf.argmax(outputLabel, 1),
                                 tf.argmax(outputLayer, 1))
            accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))
            accuracyValue = sess.run(accuracy, feed_dict={
                                     inputLayer: batch[0], outputLabel: batch[1]})
            print(i, 'train set accuracy:', accuracyValue)

# test
        corrected = tf.equal(tf.argmax(outputLabel, 1),
                             tf.argmax(outputLayer, 1))
        accuracy = tf.reduce_mean(tf.cast(corrected, tf.float32))
        accuracyValue = sess.run(accuracy, feed_dict={
                                 inputLayer: mnist.test.images, outputLabel: mnist.test.labels})
        # print("accuracy on test set:", accuracyValue)
