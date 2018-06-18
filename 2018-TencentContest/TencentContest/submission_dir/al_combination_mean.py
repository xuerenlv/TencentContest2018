# coding:utf-8

import pandas as pd
import os
import numpy as np


def pro_mean(pdir):
    pd_list, weight_list = [], []
    for fi in os.listdir(pdir):
        if 'no' in fi:
            continue
        fpth = os.path.join(pdir, fi)
        print fpth
        pd_list.append(pd.read_csv(fpth))


    p1 = pd_list[0]
    for i in range(1, len(pd_list)):
        p1['score'] += pd_list[i]['score']
    p1['score'] /= len(pd_list)
    p1.to_csv(os.path.join(pdir, 'mean_submission.csv'), index=False)





if __name__ == '__main__':
    # pdir = './pb_sub/231_s18'
    # pdir = './pb_sub/submission_pp'
    pdir = './pb_sub/submission_pp_4'
    pro_mean(pdir)
    pass






















