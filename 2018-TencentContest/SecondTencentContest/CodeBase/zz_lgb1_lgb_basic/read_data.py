# coding:utf-8

import pandas as pd
import argparse
from sklearn.utils import shuffle
import numpy as np
import gc
import os
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse


def get_prod_dataset(formal):
    if formal:
        dir_name = '../../DataSet/preliminary_contest_data/'
    else:
        dir_name = '../../DataSet/small_preliminary_contest_data/'

    pos_train_data_file = dir_name + 'finally_processed_data_train_pos.csv'
    neg_train_data_file = dir_name + 'finally_processed_data_train_neg.csv'
    predict_data_file1 = dir_name + 'finally_processed_data_predict_1.csv'
    predict_data_file2 = dir_name + 'finally_processed_data_predict_2.csv'
    user_data_file = dir_name + 'finally_processed_data_user.csv'
    ad_data_file = dir_name + 'finally_processed_data_ad.csv'
    feature_conf_dict_file = dir_name + 'finally_feature_conf_dict.pic'

    uid_map_file = dir_name + 'finally_uid_map_dict.pic'
    aid_map_file = dir_name + 'finally_aid_map_dict.pic'

    pos_train_data, neg_train_data, predict_data1, predict_data2, user_data, ad_data = pd.read_csv(pos_train_data_file), \
        pd.read_csv(neg_train_data_file), pd.read_csv(predict_data_file1), \
        pd.read_csv(predict_data_file2), pd.read_csv(user_data_file), pd.read_csv(ad_data_file)

    feature_conf_dict = pickle.load(open(feature_conf_dict_file, 'r'))
    uid_map = pickle.load(open(uid_map_file, 'r'))
    aid_map = pickle.load(open(aid_map_file, 'r'))
    return pos_train_data, neg_train_data, predict_data1, predict_data2, user_data, ad_data, feature_conf_dict, uid_map, aid_map


def get_user_data(graph_hyper_params):
    if graph_hyper_params['formal']:
        dir_name = '../../DataSet/preliminary_contest_data/'
    else:
        dir_name = '../../DataSet/small_preliminary_contest_data/'
    user_data_file = dir_name + 'finally_processed_data_user.csv'
    user_data = pd.read_csv(user_data_file)
    return user_data


def save_file(final_predict_data, predict_data_ori, train_data_ori, dev_data_ori, graph_hyper_params):
    if graph_hyper_params['formal']:
        dir_name = '../../DataSet/preliminary_contest_data/' + 'n' +str(graph_hyper_params['neg_start'])
    else:
        dir_name = '../../DataSet/small_preliminary_contest_data/' + 'n' +str(graph_hyper_params['neg_start'])

    final_predict_data_file = dir_name + '_submission.csv'
    predict_data_ori_file = dir_name + '_predict_data_leaf.csv'
    train_data_ori_file = dir_name + '_train_data_leaf.csv'
    dev_data_ori_file = dir_name + '_dev_data_leaf.csv'


    final_predict_data.to_csv(final_predict_data_file, index=False)
    predict_data_ori.to_csv(predict_data_ori_file, index=False)
    train_data_ori.to_csv(train_data_ori_file, index=False)
    dev_data_ori.to_csv(dev_data_ori_file, index=False)

    print 'submission_path:', os.path.abspath(final_predict_data_file)


if __name__ == '__main__':
    pos_train_data, neg_train_data, predict_data1, predict_data2, user_data, ad_data, feature_conf_dict, uid_map, aid_map = get_prod_dataset(False)

    print len(pos_train_data), len(neg_train_data)



    pass
