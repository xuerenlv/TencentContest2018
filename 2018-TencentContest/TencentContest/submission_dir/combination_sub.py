# coding:utf-8

import pandas as pd
import os
import numpy as np


def pro_mean(pdir, mean):
    pd_list, weight_list = [], []
    for fi in os.listdir(pdir):
        if len(fi.split('_')) < 2:
            continue
        weight_list.append(float(fi.split('_')[0]))
        fpth = os.path.join(pdir, fi)
        pd_list.append(pd.read_csv(fpth))

    if mean:
        p1 = pd_list[0]
        for i in range(1, len(pd_list)):
            p1['score'] += pd_list[i]['score']
        p1['score'] /= len(pd_list)
        p1.to_csv(os.path.join(pdir, 'mean.csv'), index=False)
    else:
        weight_list = np.array(weight_list)/sum(weight_list)
        for i in range(0, len(pd_list)):
            pd_list[i]['score'] = pd_list[i]['score'] * weight_list[i]
        p1 = pd_list[0]
        for i in range(1, len(pd_list)):
            p1['score'] += pd_list[i]['score']
        p1.to_csv(os.path.join(pdir, 'weight.csv'), index=False)
    print pdir, mean, ' done !'





if __name__ == '__main__':
    pdir = './pa_submission/231'
    pro_mean(pdir, mean=True)
    pro_mean(pdir, mean=False)
    pass






















