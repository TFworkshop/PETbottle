# -*- coding: utf-8 -*-

import numpy as np

# カテゴリテーブル
_category_table = np.array([
    'その他',
    'コーラ',
    '炭酸飲料',
    'コーヒー飲料',
    'スポーツ飲料',
    'ミネラルウォーター',
    '緑茶',
    '紅茶',
    'その他お茶',
    '果汁飲料',
    '乳性飲料'])

# カテゴリ数を取得
def get_category_num():
    return len(_category_table)

# 指定したカテゴリの番号を取得
def get_category_no(category):
    for i, cate in enumerate(_category_table):
        if cate == category:
            return i
    # 一致しない場合, その他に
    return 0 

# 指定した番号のカテゴリ表記を取得
def get_category(no):
    if no >= 0 and no < len(_category_table):
        return _category_table[no]
    return None
