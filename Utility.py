import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import time
import logging


def visualizeAugData(imgs, labels):
    gridH = 10
    gridW = 10
    fig, axs = plt.subplots(gridH, gridW)

    fig.set_size_inches(20, 20 * (gridH / gridW))

    numImgs = imgs.shape[0]
    Ids = list(range(numImgs))
    np.random.shuffle(Ids)
    print("Number corners:", numImgs)
    for iA, iC in itertools.product(range(gridH), range(gridW)):
        if iC + gridW * iA >= numImgs:
            break
        imgId = Ids[iC + gridW * iA]

        axs[iA, iC].imshow(np.squeeze(imgs[imgId, :, :]), cmap='gray')
        axs[iA, iC].set_title(str(np.argmax(labels[imgId, :])))
        axs[iA, iC].axis('off')

def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initialWeights =tf.constant(0.1,shape=shape)
	return tf.Variable(initialWeights)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def getConvLayers(inputLayer, inputChannel, convSizes, convChannels, maxPoolingPosition,
                  padding='VALID', training_ph=None, batchNorm=False, netName=''):
    ws = []
    bs = []
    assert (len(convChannels) == len(convSizes))

    for i in range(len(convChannels)):
        if i == 0:
            w = tf.get_variable(netName + "WConv%d" % i,
                                shape=[convSizes[i], convSizes[i], inputChannel, convChannels[i]],
                                initializer=tf.initializers.glorot_normal())
        else:
            w = tf.get_variable(netName + "WConv%d" % i,
                                shape=[convSizes[i], convSizes[i], convChannels[i - 1], convChannels[i]],
                                initializer=tf.initializers.glorot_normal())

        b = tf.get_variable(netName + "bConv%d" % i, shape=[convChannels[i]], initializer=tf.initializers.zeros())

        ws.append(w)
        bs.append(b)

        # Should not use drop out in CNN layers!!!
        # if i == 0:
        #     cnn = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(inputLayer/ 255, w, strides = [1,1,1,1], padding = 'VALID') + b), pkeep_ph)
        # else:
        #     cnn = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(cnn, w, strides = [1,1,1,1], padding = 'VALID') + b), pkeep_ph)

        if i == 0:
            cnn = tf.nn.conv2d(inputLayer / 255, w, strides=[1, 1, 1, 1], padding=padding) + b
        else:
            cnn = tf.nn.conv2d(cnn, w, strides=[1, 1, 1, 1], padding=padding) + b

        if batchNorm:
            cnn = tf.layers.batch_normalization(cnn, axis=[-1], training=training_ph)

        cnn = tf.nn.relu(cnn)

        if i in maxPoolingPosition:
            cnn = tf.nn.max_pool(cnn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

        print(cnn.get_shape())

    return cnn, ws, bs


def getFlattenSize(layer):
    shape = layer.get_shape()
    return shape[-1] * shape[-2] * shape[-3]

def compute_accuracy(v_xs, v_ys, prediction, sess, batchSize = 5000):
    numBatchs = int(np.ceil(v_xs.shape[0] / batchSize))

    numCorrectAll = 0
    for iBatch in range(numBatchs):
        batchData = v_xs[iBatch * batchSize:(iBatch + 1) * batchSize]
        gdData = v_ys[iBatch * batchSize:(iBatch + 1) * batchSize, :]
        # y_pre = sess.run(prediction, feed_dict={'x_image': 'v_xs', 'keep_prob': 1})
        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(gdData,1))
        numCorrect = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        result = sess.run(numCorrect, feed_dict={'x_image:0': batchData, 'ys:0': gdData, 'keep_prob:0': 1})

        numCorrectAll = numCorrectAll + result
    return numCorrectAll / v_xs.shape[0]