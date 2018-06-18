# coding:utf-8


import pandas as pd
import argparse
from sklearn.utils import shuffle
import numpy as np
import gc
import os
import cPickle as pickle

from read_data import get_prod_dataset

train_data, dev_data, predict_data, relevant_user_data, no_relevant_user_data, ad_data, feature_conf_dict, re_uid_map, re_aid_map =get_prod_dataset(formal=False)
# dir_name = '../../DataSet/preliminary_contest_data/'
# dir_name = '../../DataSet/small_preliminary_contest_data/'

# train_data_file = dir_name + 'finally_processed_data_train.csv'
# dev_data_file = dir_name + 'finally_processed_data_dev.csv'

# train_data = pd.read_csv(dev_data_file)
# dev_data = pd.read_csv(dev_data_file)


# print np.array(train_data['uid'])
da_al = pd.concat([train_data, dev_data])
pos_atd, neg_atd = da_al[da_al['label'] == 1], da_al[da_al['label'] == 0]

print relevant_user_data[]

print len(set(pos_atd['uid'])), len(pos_atd)
print len(set(neg_atd['uid'])), len(neg_atd)
















