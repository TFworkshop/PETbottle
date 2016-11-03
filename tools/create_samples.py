#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import sys
import numpy as np
import cv2

# 出力ディレクトリ
OUTPUT_DIR = './out'

# 画像変換パラメータ
# (注)比率は同じに!!
OBJ_SIZE_W = 100  # suntory 200, cocacola 40, asahi 100
OBJ_SIZE_H = 200  # suntory 400, cocacola 80, asahi 200
OUT_SIZE_W = 20   # pattern size
OUT_SIZE_H = 40   # pattern size

# 画像生成パラメータ
# 横方向にダミー付加(width * X))
MARGIN_RATIO_X = 1.8  # suntory/asahi 1.8, cocacoka 1.0
# 縦方向にダミー付加(height * Y))
MARGIN_RATIO_Y = 1.4  # suntory/asahi 1.4, cocacola 1.0

# ガンマ変換
GAM_START = 1
GAM_END   = 3
GAM_STEP  = 1
GAM_SCALE = 0.75

# 平滑化
SMT_START = 2
SMT_END   = (4 + 1)
SMT_STEP  = 2

# 回転(アフィン変換)
ROT_START = -5
ROT_END   = (5 + 1)
ROT_STEP  = 2

# 射影変換
PSX_DELTA  = 0.25
PSY_DELTA  = 0.1

# Salt&Pepperノイズ
SPN_START = 1
SPN_END   = (5 + 1)
SPN_STEP  = 2
SPN_SCALE = 0.001

def options(argv):
    files = []
    ret = True
    opt = 0
    # parse
    i = 0
    while i < len(argv):
        if argv[i] == '-cnv':
            opt = 1
        else:
            if argv[i].endswith('/'):
                cmd = argv[i]+'*.'
            else:
                cmd = argv[i]+'/*.'
            for ex in ['jpg', 'png']:
                files.extend(glob.glob(cmd+ex))
        i += 1

    if files == []:
        print('Not found image file.')
        ret = False
                        
    return ret, opt, files

# 画像貼り付け
def paste(dst, src, x, y, width, height):
    resize = cv2.resize(src, tuple([width, height]))
    if x >= dst.shape[1] or y >= dst.shape[0]:
        return None
    if x >= 0:
        w = min(dst.shape[1] - x, resize.shape[1])
        u = 0
    else:
        w = min(max(resize.shape[1] + x, 0), dst.shape[1])
        u = min(-x, resize.shape[1] - 1)
    if y >= 0:
        h = min(dst.shape[0] - y, resize.shape[0])
        v = 0
    else:
        w = min(max(resize.shape[0] + y, 0), dst.shape[0])
        v = min(-y, resize.shape[0] - 1)
    dst[y:y+h, x:x+w] = resize[v:v+h, u:u+w]
    return dst

# オリジナル
def original(src):
    # 背景画像生成
    height,width = src.shape[:2]
    base_width = int(width * MARGIN_RATIO_X)
    base_height = int(height * MARGIN_RATIO_Y)
    size = tuple([base_height, base_width, 3])
    base_img = np.zeros(size, dtype=np.uint8)
    base_img.fill(255)
    # 画像貼り付け
    sx = int(width * (MARGIN_RATIO_X - 1.0) / 2)
    sy = int(height * (MARGIN_RATIO_Y - 1.0) / 2)
    return paste(base_img, src, sx, sy, width, height)

# ガンマ変換
# 係数：0.75, 1.5
def gamma(src):
    samples = []
    LUTs = []
    for g in range(GAM_START, GAM_END, GAM_STEP):
        LUT_G = np.arange(256, dtype = 'uint8' )
        for i in range(256):        
            LUT_G[i] = 255 * pow(float(i) / 255, 1.0 / (g * GAM_SCALE)) 
        LUTs.append(LUT_G)
    for i, LUT in enumerate(LUTs):
        samples.append(cv2.LUT(src, LUT))
    return samples

# 平滑化(Image Blurring)
# 係数：2x2, 4x4
def smoothing(src):
    samples = []
    for size in range(SMT_START,SMT_END,SMT_STEP):
        samples.append(cv2.blur(src, (size, size)))
    return samples

# 回転(アフィン変換)
# 係数：angle = -5 - 5, step 2
def rotation(src):
    samples = []
    # size(width, height)
    size = tuple(np.array([src.shape[1], src.shape[0]]))
    center = tuple(np.array([src.shape[1]/2, src.shape[0]/2]))
    for angle in range(ROT_START,ROT_END,ROT_STEP):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        samples.append(cv2.warpAffine(src, rotation_matrix, 
                                      size, flags=cv2.INTER_CUBIC))
    return samples

# 射影変換
def perspective(src):
    samples = []
    # size(width, height)
    size = tuple(np.array([src.shape[1], src.shape[0]]))
    center = tuple(np.array([src.shape[1]/2, src.shape[0]/2]))
    if center[0] > 12 and center[1] > 24:
        po_src = np.float32([[center[0]-10, center[1]-20], 
                             [center[0]+10, center[1]-20],
                             [center[0]-10, center[1]+20],
                             [center[0]+10, center[1]+20]])
        # top 9.75 / bottom 10
        po_dst = po_src.copy()
        po_dst[0][0] += PSX_DELTA
        po_dst[1][0] -= PSX_DELTA
        ps_matrix = cv2.getPerspectiveTransform(po_src, po_dst)
        samples.append(cv2.warpPerspective(src, ps_matrix, size))

        # top 10 / bottom 9.75
        po_dst = po_src.copy()
        po_dst[2][0] += PSX_DELTA
        po_dst[3][0] -= PSX_DELTA
        ps_matrix = cv2.getPerspectiveTransform(po_src, po_dst)
        samples.append(cv2.warpPerspective(src, ps_matrix, size))

        # left 19.9 / right 21
        po_dst = po_src.copy()
        po_dst[0][1] += PSY_DELTA
        po_dst[2][1] -= PSY_DELTA
        ps_matrix = cv2.getPerspectiveTransform(po_src, po_dst)
        samples.append(cv2.warpPerspective(src, ps_matrix, size)) 

        # left 21 / right 19.9
        po_dst = po_src.copy()
        po_dst[1][1] += PSY_DELTA
        po_dst[3][1] -= PSY_DELTA
        ps_matrix = cv2.getPerspectiveTransform(po_src, po_dst)
        samples.append(cv2.warpPerspective(src, ps_matrix, size))
    return samples

# Salt&Pepperノイズ
# 係数：amount = 0.001 - 0.005, step 0.002)
def saltpepper_noise(src):
    row, cal, ch = src.shape
    samples = []
    s_vs_p = 0.5
    for amount in range(SPN_START,SPN_END,SPN_STEP):
        out = src.copy()
        # Salt mode
        num_salt = np.ceil(amount * SPN_SCALE * src.size * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in src.shape]
        out[coords[:-1]] = (255, 255, 255)
        # Pepper mode
        num_pepper = np.ceil(amount * SPN_SCALE * src.size * (1-s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in src.shape]        
        out[coords[:-1]] = (0, 0, 0)
        samples.append(out)
    return samples

# 生成サンプル保存
def save_samples(dir, opt, file, samples):
    if not os.path.exists(dir):
        os.mkdir(dir)

    base = os.path.splitext(os.path.basename(file))[0] + '_'
    for i, img in enumerate(samples):
        if opt == 1:
            size = str(OUT_SIZE_W) + 'x' + str(OUT_SIZE_H) + '_'
            out = os.path.join(dir, base+size+str(i)+'.jpg')
            print(out)
            # オブジェクト領域抽出
            sx = (img.shape[1] - OBJ_SIZE_W) / 2
            sy = (img.shape[0] - OBJ_SIZE_H) / 2
            ex = (img.shape[1] + OBJ_SIZE_W) / 2
            ey = (img.shape[0] + OBJ_SIZE_H) / 2
            img1 = img[sy:ey, sx:ex]
            # リサイズ
            img2 = cv2.resize(img1, (OUT_SIZE_W, OUT_SIZE_H))
            cv2.imwrite(out, img2)

        else:
            out = os.path.join(dir, base+str(i)+'.jpg')
            print(out)
            cv2.imwrite(out, img)

def main(opt, files):
    for file in files:
        base_samples = []
        samples = []

        # イメージファイル読み込み
        src_img = cv2.imread(file)
        # 周囲拡張
        src_img = original(src_img)
        base_samples.append(src_img)
        # 回転
        base_samples.extend(rotation(src_img))
        # 射影
        base_samples.extend(perspective(src_img))
        # 平滑化, ノイズ付加等
        for img in base_samples:
            samples.append(img)
            # ガンマ変換
            # 係数：0.75, 1.5
            samples.extend(gamma(img))
            # 平滑化(Image Blurring)
            # 係数：2x2, 4x4
            samples.extend(smoothing(img))
            # Salt&Pepperノイズ
            # 係数：amount = 0.001 - 0.006, step 0.002)
            samples.extend(saltpepper_noise(img))
        save_samples(OUTPUT_DIR, opt, file, samples)
        
if __name__ == '__main__':
    ret, opt, files = options(sys.argv[1:])
    if ret == False:
        sys.exit('Usage: %s <-cnv> input_directory' % sys.argv[0])
    main(opt, files)
