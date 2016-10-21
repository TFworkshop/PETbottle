#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import sys
import numpy as np
import cv2
import selectivesearch

SS_IMG_SIZE = 100

def options(argv):
    files = None
    ret = True
    if len(argv) == 1:
        # files = os.listdir(argv[i])
        cmd = argv[0]+'/*.jpg'
        files = glob.glob(cmd)
        if files == None:
            ret = False
    else:
        ret = False
    return ret, files

def resize_image(src_img):
    # 縮小比率計算
    img_height, img_width = src_img.shape[:2]
    w_ratio = int(img_width / SS_IMG_SIZE)
    h_ratio = int(img_height / SS_IMG_SIZE)
    if w_ratio > h_ratio:
        ratio = h_ratio
    else:
        ratio = w_ratio
    if ratio < 1:
        ratio = 1
    size = (img_height/ratio, img_width/ratio)
    # 画像サイズを縮小(高速化のため)
    img = cv2.resize(src_img, size)
    return img, ratio

def check_regions(regions):
    cands = set()
    # 検知したペットボトルを矩形で囲む
    for r in regions:
        if r['rect'] in cands:
            continue
        if r['size'] < 1000:
            continue
        x, y, w, h = r['rect']
        if w == 0:
            continue
        if h / w < 2.0 or h / w > 5.0:
            continue
        cands.add(r['rect'])
    return cands
    
def main(files):
    for img_file in files:
        # イメージファイル読み込み
        src_img = cv2.imread(img_file)
        # 画像縮小
        img, ratio = resize_image(src_img)
        # selectivesearchによる領域検出
        label, regions = selectivesearch.selective_search(
            img, scale=100, sigma=0.9, min_size=10)
        cands = check_regions(regions)
        # 検知したペットボトルを矩形で囲む
        if len(cands) == 0:
            print("%s  %d  "%(img_file, len(cands)))
        else:
            print("%s  %d  "%(img_file, len(cands))),
            for r in cands:
                x, y, w, h = [i*ratio for i in r]
                cv2.rectangle(src_img,(x,y),(x+w,y+h),(255,0,0),2)
                print(" %d %d %d %d"%(x, y, w, h)),
                res_file = 'res_' + os.path.basename(img_file)
                cv2.imwrite(res_file, src_img)
            print('')

if __name__ == '__main__':
    ret, files = options(sys.argv[1:])
    if ret == False:
        sys.exit('Usage: %s image_directory'%sys.argv[0])
    main(files)
