#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import sys
import numpy as np
import cv2

def options(argv):
    i = 0
    files = None
    cascade = None
    ret = True
    while i < len(argv):
        if argv[i] == '-c':
            cascade = cv2.CascadeClassifier(argv[i+1])
            i+=2
        else:
            # files = os.listdir(argv[i])
            cmd = argv[i]+'/*.jpg'
            print cmd
            files = glob.glob(cmd)
            i+=1
    if files == None or cascade == None:
        ret = False
    return ret, files, cascade

def main(files, cascade):
    for img_file in files:
        # イメージファイル読み込み
        src_img = cv2.imread(img_file)
        # グレースケール化
        gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        # ペットボトル検出
        bottles = cascade.detectMultiScale(gray_img, 1.1, 3)
        if len(bottles) == 0:
            print("%s  %d  "%(img_file, len(bottles)))
        else:
            # 検知したペットボトルを矩形で囲む
            print("%s  %d  "%(img_file, len(bottles))),
            for (x,y,w,h) in bottles:
                cv2.rectangle(src_img,(x,y),(x+w,y+h),(255,0,0),2)
                print("%d %d %d %d "%(x, y, w, h))
    
            res_file = 'res_' + os.path.basename(img_file)
            cv2.imwrite(res_file, src_img)

if __name__ == '__main__':
    ret, files, cascade = options(sys.argv[1:])
    if ret == False:
        sys.exit('Usage: %s <-c cascade file>  directory'%sys.argv[0])
    main(files, cascade)
