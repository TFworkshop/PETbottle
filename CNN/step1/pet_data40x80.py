# -*- coding: utf-8 -*-

import csv
import numpy as np
from PIL import Image
import pet_info as pinfo

IMAGE_WIDTH  = 40
IMAGE_HEIGHT = 80
IMAGE_DEPTH  = 3

DATA_MAX     = 50000

class PETBottle:
    def __init__(self, LT):
        if LT == 'L':
            self.fnList = './train_data/train_40x80_all.csv'
        else:
            self.fnList = './train_data/test_40x80.csv'

    def getImage(self):
        return _read_image(self.fnList)

# 画像データ読み込み
def _read_image(fnList):
    size = IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_DEPTH
    images = np.zeros((DATA_MAX, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    labels = np.zeros(DATA_MAX, dtype=np.uint8)
    f = open(fnList, 'rb')
    reader = csv.reader(f)
    i = 0
    for row in reader:
        if (i >= DATA_MAX):
            print('Data buffer overflow.')
            break

        labels[i] = pinfo.get_category_no(row[1])
        byte_buffer = np.array(Image.open(row[0]), dtype=np.uint8)
        images[i] = byte_buffer.astype(np.float32) / 255.0
        i += 1

    f.close()
    return i, images, labels
