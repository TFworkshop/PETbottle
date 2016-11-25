# -*- coding: utf-8 -*-

import argparse
import tools.find_mxnet
import mxnet as mx
import os
import importlib
import sys
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from detect.detector import Detector
from PIL import Image
import tensorflow as tf
import random

import pet_eval as peval
import pet_info as pinfo

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

OUT_IMG_WIDTH  = 40
OUT_IMG_HEIGHT = 80
PET_THRESH     = 0.5

sess = tf.Session()
images = tf.placeholder(tf.float32, shape=[None, peval.IMAGE_HEIGHT, peval.IMAGE_WIDTH, peval.IMAGE_DEPTH])
logits = peval.inference(images)
saver = tf.train.Saver()
saver.restore(sess, peval.PET_CNN_MODEL)

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx,
                 nms_thresh=0.5, force_nms=True):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    force_nms : bool
        force suppress different categories
    """
    sys.path.append(os.path.join(os.getcwd(), 'symbol'))
    net = importlib.import_module("symbol_" + net) \
        .get_symbol(len(CLASSES), nms_thresh, force_nms)
    detector = Detector(net, prefix + "_" + str(data_shape), epoch, \
        data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        choices=['vgg16_reduced'], help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/dog.jpg',
                        help='run demo with images, use comma(without space) to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd'), type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.5,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    args = parser.parse_args()
    return args

def disp_result(res):
    print('--------------------------')
    print res
    for i in range(3):
        idx = np.argsort(res)[0][::-1][i]
        print('%d位 %s (%f)' % ((i+1), pinfo.get_category(idx), np.sort(res)[0][::-1][i]))
        if i == 0:
            first = pinfo.get_category_en(idx)
            score = np.sort(res)[0][::-1][i]
    return first, score

def bottle_recognition(img, dets, thresh=0.6):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(dets.shape[0]):
        cls_id = int(dets[i, 0])
        score = dets[i, 1]
        if cls_id == 4 and score > PET_THRESH:
            xmin = int(det[i, 2] * width)
            ymin = int(det[i, 3] * height)
            xmax = int(det[i, 4] * width)
            ymax = int(det[i, 5] * height)
            # ペットボトル領域表示
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            # 切り出した領域の認識
            area = img[ymin:ymax, xmin:xmax]
            resize = cv2.resize(area, (OUT_IMG_WIDTH, OUT_IMG_HEIGHT))
            resize[:, :, (0, 1, 2)] = resize[:, :, (2, 1, 0)] # OpenCV B/G/R (IMageとは逆) 
            inps = resize.astype(np.float32) / 255.0
            inps = sess.run(tf.expand_dims(tf.cast(inps, tf.float32), 0))
            cand = sess.run(logits, feed_dict={images: inps})
            res, score = disp_result(cand)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if (ymin - 30) < 0:
                ybase = ymin + 35
            else:
                ybase = ymin
            cv2.putText(img, res, (xmin+3, ybase-20), font, 0.4, (0,0,255))
            cv2.putText(img, str(score), (xmin+3, ybase-8), font, 0.4, (0,0,255))
    # 入力画像表示
    cv2.imshow('Input Image', img)
    cv2.imwrite('tmp01.jpg', img) 
    cv2.waitKey(0)

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.cpu()
    # parse image list
    image_list = [i.strip() for i in args.images.split(',')]
    assert len(image_list) > 0, "No valid image specified to detect"

    detector = get_detector(args.network, args.prefix, args.epoch,
                            args.data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx, args.nms_thresh, args.force_nms)
    # run detection
    dets = detector.im_detect(image_list, args.dir, args.extension,
                                  args.show_timer)
    for k, det in enumerate(dets):
        img = cv2.imread(image_list[k])
        bottle_recognition(img, det, PET_THRESH)
