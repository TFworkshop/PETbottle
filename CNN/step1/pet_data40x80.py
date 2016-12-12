# -*- coding: utf-8 -*-

import sys
import csv
import random
import numpy as np
from PIL import Image
import pet_info as pinfo
import tensorflow as tf

IMAGE_WIDTH  = 32
IMAGE_HEIGHT = 64
IMAGE_DEPTH  = 3

DATA_MAX     = 35000

class PETBottle:
    def __init__(self, LT):
        if LT == 'L':
            self.fnList = './train_data/train_40x80_all.csv'
        else:
            self.fnList = './train_data/test_40x80.csv'
        _read_datlist(self)

    def getRandomImages(self, datnum):
        return _get_random_images(self, datnum)

    def getImage(self, idx):
        return _get_image(self, idx)
    
    def getDatNum(self):
        return self.datnum

# データリストの読み込み
def _read_datlist(self):
    f = open(self.fnList, 'rb')
    reader = csv.reader(f)
    datlst = []
    i = 0
    for row in reader:
        datlst.append(row)
        i += 1
    f.close()
    self.datlst = datlst
    self.datnum = i

# 画像データ読み込み(ランダムにN個)
def _get_random_images(self, datnum):
    images = np.zeros((datnum, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    labels = np.zeros(datnum, dtype=np.uint8)
    idxs = random.sample(xrange(len(self.datlst)), datnum)
    i = 0
    for idx in idxs:
        row = self.datlst[idx]
        labels[i] = pinfo.get_category_no(row[1])
        tmpimg = Image.open(row[0])
        x = random.randint(0,7)
        y = random.randint(0,15)
        tmpimg = tmpimg.crop((x, y, x+IMAGE_WIDTH, y+IMAGE_HEIGHT))
        # images[i] = np.array(tmpimg, np.float32) / 255.0
        images[i] = _image_whitening(np.array(tmpimg, np.float32))
        i += 1
    return images, labels

# 画像データ読み込み(N番目)
def _get_image(self, idx):
    if idx >= self.datnum:
        return None, None
    row = self.datlst[idx]
    label = pinfo.get_category_no(row[1])
    tmpimg = Image.open(row[0])
    tmpimg = tmpimg.crop((3, 7, 3+IMAGE_WIDTH, 7+IMAGE_HEIGHT))
    # image = np.array(tmpimg, np.float32) / 255.0
    image = _image_whitening(np.array(tmpimg, np.float32))
    return image, label

# ホワイトニング処理
def _image_whitening(img):
    # img = img.astype(np.float32)
    d, w, h = img.shape
    num_pixels = d * w * h
    mean = img.mean()
    variance = np.mean(np.square(img)) - np.square(mean)
    stddev = np.sqrt(variance)
    min_stddev = 1.0 / np.sqrt(num_pixels)
    scale = stddev if stddev > min_stddev else min_stddev
    img -= mean
    img /= scale
    # print img
    return img

