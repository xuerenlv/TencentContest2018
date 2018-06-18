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
    train_data_file = dir_name + 'finally_processed_data_train.csv'
    dev_data_file = dir_name + 'finally_processed_data_dev.csv'
    predict_data_file = dir_name + 'finally_processed_data_predict.csv'
    relevant_user_data_file = dir_name + 'finally_processed_data_user_relevant.csv'
    no_relevant_user_data_file = dir_name + 'finally_processed_data_user_no_rel.csv'
    ad_data_file = dir_name + 'finally_processed_data_ad.csv'
    feature_conf_dict_file = dir_name + 'finally_feature_conf_dict.pic'

    re_uid_map_file = dir_name + 'finally_re_uid_map_dict.pic'
    re_aid_map_file = dir_name + 'finally_re_aid_map_dict.pic'

    train_data, dev_data, predict_data, relevant_user_data, no_relevant_user_data, ad_data = pd.read_csv(train_data_file), \
        pd.read_csv(dev_data_file), pd.read_csv(predict_data_file), \
        pd.read_csv(relevant_user_data_file), pd.read_csv(no_relevant_user_data_file), pd.read_csv(ad_data_file)

    feature_conf_dict = pickle.load(open(feature_conf_dict_file, 'r'))
    re_uid_map = pickle.load(open(re_uid_map_file, 'r'))
    re_aid_map = pickle.load(open(re_aid_map_file, 'r'))
    return train_data, dev_data, predict_data, relevant_user_data, no_relevant_user_data, ad_data, feature_conf_dict, re_uid_map, re_aid_map


def pro_vec_fea(x):
    if type(x) != str or '|' not in x:
        return ''
    return x.split('|')[0]


def cluseter(formal):
    if formal:
        dir_name = '../../DataSet/preliminary_contest_data/'
    else:
        dir_name = '../../DataSet/small_preliminary_contest_data/'
    relevant_user_data_file = dir_name + 'finally_processed_data_user_relevant.csv'
    no_relevant_user_data_file = dir_name + 'finally_processed_data_user_no_rel.csv'
    user_data = pd.concat([pd.read_csv(relevant_user_data_file), pd.read_csv(no_relevant_user_data_file)])

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education',
                       'gender', 'house', 'os', 'ct', 'marriageStatus']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
                      'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',
                      'topic1', 'topic2', 'topic3']

    cv = CountVectorizer()
    for fea in vector_feature:
        user_data[fea] = user_data[fea].apply(pro_vec_fea)
        cv.fit(user_data[fea])

    # print user_data['appIdInstall']
    print user_data



    pass


if __name__ == '__main__':
    cluseter(False)
    pass


