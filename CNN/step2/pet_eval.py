# -*- coding: utf-8 -*-

import os
import sys
import struct
import random
import numpy as np
import tensorflow as tf

IMAGE_WIDTH  = 32
IMAGE_HEIGHT = 64
IMAGE_DEPTH  = 3

BOTTLE_CATEGORY_SIZE = 11

CANDIDATE_MAX = 5

CONV1_FILTER_NUM = 64
CONV2_FILTER_NUM = 128
FULL_CONNECT_NUM = 256

PET_CNN_MODEL    = './cnn_model/v31_40x80_20000-20000'

# 重み&バイアス初期化関数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution & Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], 
                          strides=[1,2,2,1], padding='SAME')

def inference(images):
    # 1層畳み込み層
    W_conv1 = weight_variable([5,5,3,CONV1_FILTER_NUM])
    b_conv1 = bias_variable([CONV1_FILTER_NUM])
    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)

    W_conv15 = weight_variable([3,3,CONV1_FILTER_NUM,CONV1_FILTER_NUM])
    b_conv15 = bias_variable([CONV1_FILTER_NUM])
    h_conv15 = tf.nn.relu(conv2d(h_conv1, W_conv15) + b_conv15)

    h_pool1 = max_pool_2x2(h_conv15)

    # 2層畳み込み層
    W_conv2 = weight_variable([3,3,CONV1_FILTER_NUM,CONV2_FILTER_NUM])
    b_conv2 = bias_variable([CONV2_FILTER_NUM])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 密に結合された層
    W_fc1 = weight_variable([8*16*CONV2_FILTER_NUM, FULL_CONNECT_NUM])
    b_fc1 = bias_variable([FULL_CONNECT_NUM])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*16*CONV2_FILTER_NUM])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # ドロップアウト
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)

    # 読み出し層
    W_fc2 = weight_variable([FULL_CONNECT_NUM, BOTTLE_CATEGORY_SIZE])
    b_fc2 = bias_variable([BOTTLE_CATEGORY_SIZE])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv

