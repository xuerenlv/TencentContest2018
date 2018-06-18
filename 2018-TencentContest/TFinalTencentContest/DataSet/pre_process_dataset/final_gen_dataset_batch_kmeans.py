# coding:utf-8

import argparse
from sklearn.utils import shuffle
import gc
import os
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
from datetime import datetime

one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education',
                   'gender', 'house', 'os', 'ct']
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
                  'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'marriageStatus']


def pro_vec_fea(x):
    if type(x) != str or '|' not in x:
        return ''
    return x.split('|')[0]


def pro_one_user(tu, start_feature_conf_dict):
    global one_hot_feature, vector_feature
    col = []
    for fea in one_hot_feature:
        if int(tu[fea]) != 0:  # 缺失值
            col.append(start_feature_conf_dict[fea] + int(tu[fea]))
    for fea in vector_feature:
        for i in pro_vec_fea(tu[fea]).split(' '):
            if len(i) == 0:
                continue
            col.append(start_feature_conf_dict[fea] + int(i))
    return col


def cluseter(formal, cl_num=None):
    if formal:
        barrer = 20000000
        dir_name = '../../DataSet/preliminary_contest_data/'

        if cl_num == 0:
            cluseter_num_name = ['cl1', 'cl2'] # , 'cl7', 'cl8', 'cl9', 'cl10']
            cluseter_num_val = [200, 400]
        elif cl_num == 1:
            cluseter_num_name = ['cl3', 'cl4']  # , 'cl7', 'cl8', 'cl9', 'cl10']
            cluseter_num_val = [800, 1000]
        elif cl_num == 3:
            cluseter_num_name = ['cl200', 'cl500', 'cl1000']  # , 'cl7', 'cl8', 'cl9', 'cl10']
            cluseter_num_val = [200, 500, 1000]
    else:
        barrer = 30
        dir_name = '../../DataSet/small_preliminary_contest_data/'
        cluseter_num_name = ['cl1', 'cl2']
        cluseter_num_val = [10, 20]
    print cluseter_num_name, cluseter_num_val
    user_data_file = dir_name + 'finally_processed_data_user.csv'
    feature_conf_dict_file = dir_name + 'finally_feature_conf_dict.pic'

    uid_list_train = []
    uid_list_other = []
    all_feature_file_train = dir_name + 'feature_coo_matrix_train.npz'
    all_feature_file_other = dir_name + 'feature_coo_matrix_other.npz'
    uid_list_file_train = dir_name + 'kmeans_uid_list_train.pic'
    uid_list_file_other = dir_name + 'kmeans_uid_list_other.pic'
    print 'read data start !'
    if not os.path.exists(all_feature_file_train):
        user_data = pd.read_csv(user_data_file)
        feature_conf_dict = pickle.load(open(feature_conf_dict_file, 'r'))
        start_feature_conf_dict, ss = {}, 0
        for fea in one_hot_feature:
            start_feature_conf_dict[fea] = ss
            ss += feature_conf_dict[fea]
        for fea in vector_feature:
            start_feature_conf_dict[fea] = ss
            ss += feature_conf_dict[fea][0]
        user_data = shuffle(user_data)
        user_data_train = user_data.iloc[:barrer, :]
        user_data_other = user_data.iloc[barrer:, :]
        del user_data
        gc.collect()
        user_data_train_len, user_data_other_len = len(user_data_train), len(user_data_other)

        all_col_list_train, all_row_list_train = [], []
        for i in tqdm(range(user_data_train_len)):
            tu = user_data_train.iloc[i]
            col_list = pro_one_user(tu, start_feature_conf_dict)
            uid_list_train.append(tu['uid'])
            all_col_list_train.extend(col_list)
            all_row_list_train.extend([i] * len(col_list))
        all_feature_train = coo_matrix(([1.0] * len(all_col_list_train), (all_row_list_train, all_col_list_train)), shape=(user_data_train_len, ss))
        sparse.save_npz(all_feature_file_train, all_feature_train)
        pickle.dump(uid_list_train, open(uid_list_file_train, 'w'))
        del all_row_list_train, all_col_list_train, all_feature_train, user_data_train
        gc.collect()

        all_col_list_other, all_row_list_other = [], []
        for i in tqdm(range(user_data_other_len)):
            tu = user_data_other.iloc[i]
            uid_list_other.append(tu['uid'])
            all_col_list_other.extend(col_list)
            all_row_list_other.extend([i] * len(col_list))

        all_feature_other = coo_matrix(([1.0] * len(all_col_list_other), (all_row_list_other, all_col_list_other)), shape=(user_data_other_len, ss))
        sparse.save_npz(all_feature_file_other, all_feature_other)
        pickle.dump(uid_list_other, open(uid_list_file_other, 'w'))
        del all_row_list_other, all_col_list_other, all_feature_other, user_data_other
        gc.collect()

    all_feature_train = sparse.load_npz(all_feature_file_train)
    all_feature_other = sparse.load_npz(all_feature_file_other)
    uid_list_train = pickle.load(open(uid_list_file_train, 'r'))
    uid_list_other = pickle.load(open(uid_list_file_other, 'r'))



    # print 'chage dtype of coomatrix'
    # all_feature = coo_matrix(all_feature, dtype=np.int)

    # del user_data; gc.collect()
    # print 'row, col', len_user_data, ss
    if formal:
        batch_size = 100000
    else:
        batch_size = 50

    print 'start clustering!'

    for i in range(len(cluseter_num_val)):
        print 'start: ', datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), cluseter_num_name[i], cluseter_num_val[i]
        clf = MiniBatchKMeans(
            n_clusters=cluseter_num_val[i],
            max_iter=10,
            batch_size=batch_size,
            verbose=1,
            n_init=3,
            compute_labels=False,
        )
        print '\t', 'batch_size:', batch_size
        clf.fit(all_feature_train)

        y_pred_train = clf.predict(all_feature_train)
        y_pred_other = clf.predict(all_feature_other)
        print 'end--: ', datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), cluseter_num_name[i], cluseter_num_val[i]
        d_k_map = dict(zip(uid_list_train + uid_list_other, list(y_pred_train) + list(y_pred_other)))

        print 'start save'
        user_data = pd.read_csv(user_data_file)
        user_data[cluseter_num_name[i]] = user_data['uid'].apply(lambda x: d_k_map[x])
        file_name = dir_name+'finally_processed_data_all_user_clusters_'+'_'.join(cluseter_num_name[0:i+1])+'.csv'
        user_data.to_csv(file_name, index=False)
        print 'save_to:', file_name



if __name__ == '__main__':
    # cluseter(False)
    # cluseter(True, cl_num=0)
    # cluseter(True, cl_num=1)
    cluseter(True, cl_num=3)
    pass



