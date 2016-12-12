# -*- coding: utf-8 -*-

import os
import sys
import struct
import numpy as np
import tensorflow as tf
import pet_data40x80 as pdat
import pet_info as pinfo
import pet_result as pres

IMAGE_WIDTH  = 32
IMAGE_HEIGHT = 64
IMAGE_DEPTH  = 3

BOTTLE_CATEGORY_SIZE = 11

ITELATION_NUM = 20000
MINI_BATCH_SIZE = 100
CANDIDATE_MAX = 5
LOG_DIR = './log'

CONV1_FILTER_NUM = 64
CONV2_FILTER_NUM = 128
FULL_CONNECT_NUM = 256

def options(argv):
    # init
    mode = 'LT'
    model = '';
    # parse
    i = 0
    while i < len(argv):
        if argv[i] == '-mdl':
            model = argv[i+1]
            i+=2

        elif argv[i] == '-mode':
            if argv[i+1] == 'T' or argv[i+1] == 'L' or argv[i+1] == 'LT':
                mode = argv[i+1]
                i+=2
            else:
                return false, mode, model
        else:
            return False, mode, model
    return True, mode, model

# model
def get_modelname(model):
    if model == 'auto':
        if tf.train.get_checkpoint_state('./'):
            ckpt = tf.train.get_checkpoint_state('./')
            last_model = ckpt.model_checkpoint_path
            return last_model
    else:
        return model

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

def main(mode, model):
    # 特徴ベクトルx[][Height][Width][Depth] 
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
    # 目標
    y_ = tf.placeholder(tf.float32, shape=[None, BOTTLE_CATEGORY_SIZE])

    # 1層畳み込み層
    W_conv1 = weight_variable([5,5,3,CONV1_FILTER_NUM])
    b_conv1 = bias_variable([CONV1_FILTER_NUM])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # 1.5層畳み込み層
    W_conv15 = weight_variable([3,3,CONV1_FILTER_NUM, CONV1_FILTER_NUM])
    b_conv15 = bias_variable([CONV1_FILTER_NUM])
    h_conv15 = tf.nn.relu(conv2d(h_conv1, W_conv15) + b_conv15)

    h_pool1 = max_pool_3x3(h_conv15)

    # 2層畳み込み層
    W_conv2 = weight_variable([3,3,CONV1_FILTER_NUM,CONV2_FILTER_NUM])
    b_conv2 = bias_variable([CONV2_FILTER_NUM])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_3x3(h_conv2)

    # 密に結合された層
    W_fc1 = weight_variable([8*16*CONV2_FILTER_NUM, FULL_CONNECT_NUM])
    b_fc1 = bias_variable([FULL_CONNECT_NUM])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*16*CONV2_FILTER_NUM])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # ドロップアウト
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 読み出し層
    W_fc2 = weight_variable([FULL_CONNECT_NUM, BOTTLE_CATEGORY_SIZE])
    b_fc2 = bias_variable([BOTTLE_CATEGORY_SIZE])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Train
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
    ce_summary = tf.scalar_summary('cross_entropy', cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.scalar_summary('accuracy', accuracy)

    # Saver
    saver = tf.train.Saver()

    # セッション
    sess = tf.Session()
    # ログ出力準備
    merged = tf.merge_all_summaries()
    summ_writer = tf.train.SummaryWriter(LOG_DIR, sess.graph_def)
    # 初期化
    sess.run(tf.initialize_all_variables())

    # 学習パラメータの復元
    if model != '':
        if model == 'auto':
            if tf.train.get_checkpoint_state('./'):
                ckpt = tf.train.get_checkpoint_state('./')
                last_model = ckpt.model_checkpoint_path
                saver.restore(sess, last_model)
        elif os.path.isfile(model):
            saver.restore(sess, model)

    if mode == 'L' or mode == 'LT' or model == '':
        print('--- Start Learning ---')
        train = pdat.PETBottle('L')
        for i in range(ITELATION_NUM):
            # ミニバッチのデータ
            batch_xs, label1 = train.getRandomImages(MINI_BATCH_SIZE)
            batch_ys = np.zeros((MINI_BATCH_SIZE, BOTTLE_CATEGORY_SIZE))
            for j in range(MINI_BATCH_SIZE):
                batch_ys[j][label1[j]] = 1

            # small batch
            if i%100 == 0:
                summary, train_acc = sess.run([merged, accuracy], 
                                              feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                summ_writer.add_summary(summary, i)
                print('step %d, training accuracy %g' % (i, train_acc))
                sys.stdout.flush()
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            if (i + 1) % 1000 == 0 and model != '':
                saver.save(sess, get_modelname(model), global_step=i+1)

    # 評価
    if mode == 'T' or mode == 'LT':
        test = pdat.PETBottle('T')
        print('--- Start Evaluating ---')
        num2 = test.getDatNum()
        test_res = pres.Accuracy(CANDIDATE_MAX)
        for i in range(num2):
            img, code = test.getImage(i)
            label2 = np.zeros((1,BOTTLE_CATEGORY_SIZE))
            label2[0][code] = 1
            cand = sess.run(y_conv, feed_dict={x: [img], y_: label2, keep_prob: 1.0})
            test_res.setResult(i, cand, code)
        test_res.calcAccuracy(num2)
        test_res.dispConfusionMatrix()

    # セッションクローズ
    sess.close()

if __name__ == '__main__':
    ret, mode, model = options(sys.argv[1:])
    if ret == False:
        sys.exit('Usage: %s <-mode L/T/LT> <-mdl model file>' %sys.argv[0])
    main(mode, model)
