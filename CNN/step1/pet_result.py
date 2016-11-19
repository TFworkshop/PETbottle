# -*- coding: utf-8 -*-

import numpy as np
import pet_info as pinfo

class Accuracy:
    def __init__(self, order):
        cate_num = pinfo.get_category_num()
        self.rank = np.zeros(order)
        self.order = order
        self.confmatrix = np.zeros((cate_num, cate_num), dtype=np.int32)

    def setResult(self, no, res, label):
        return _setResult(self, no, res, label)

    def calcAccuracy(self, datnum):
        return _calcAccuracy(self, datnum)

    def dispConfusionMatrix(self):
        _dispConfusionMatrix(self)

def _setResult(self, no, res, label):
    correct = pinfo.get_category(label)
    print('--- No. %5d  %s ---' % ((no+1), correct))
    for i in range(self.order):
        idx = np.argsort(res)[0][::-1][i]
        cand = pinfo.get_category(idx)
        # Make Matrix
        if i == 0:
            self.confmatrix[label][idx] += 1
        if idx == label:
            self.rank[i] += 1
            print('* %d位 : %s (%f)' % ((i+1), cand,
                                      np.sort(res)[0][::-1][i]))
        else:
            print('  %d位 : %s (%f)' % ((i+1), cand,
                                      np.sort(res)[0][::-1][i]))

def _calcAccuracy(self, datnum):
    print ('---------- Total Accuracy ----------')
    acc = self.rank[0] * 100 / datnum
    for i in range(self.order):
        print('  %d位 : %6.2f %% ( %5d / %5d )' % ((i+1), 
                      (self.rank[i] * 100 / datnum), self.rank[i], datnum))
        if i < 4:
            self.rank[i+1] += self.rank[i]
    return acc

def _dispConfusionMatrix(self):
    num = pinfo.get_category_num()
    print('Category'),
    for i in range(num):
        print(',%s'% pinfo.get_category(i)),
    print('')
    for j in range(num):
        print('%s' % pinfo.get_category(j)),
        for i in range(num):
            print(',%d' % self.confmatrix[j][i]),
        print('')

