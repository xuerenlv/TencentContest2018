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
from sklearn.cluster import KMeans
from tqdm import tqdm
from datetime import datetime


# def get_prod_dataset(formal):
#     if formal:
#         dir_name = '../../DataSet/preliminary_contest_data/'
#     else:
#         dir_name = '../../DataSet/small_preliminary_contest_data/'
#     train_data_file = dir_name + 'finally_processed_data_train.csv'
#     dev_data_file = dir_name + 'finally_processed_data_dev.csv'
#     predict_data_file = dir_name + 'finally_processed_data_predict.csv'
#     relevant_user_data_file = dir_name + 'finally_processed_data_user_relevant.csv'
#     no_relevant_user_data_file = dir_name + 'finally_processed_data_user_no_rel.csv'
#     ad_data_file = dir_name + 'finally_processed_data_ad.csv'
#     feature_conf_dict_file = dir_name + 'finally_feature_conf_dict.pic'
#
#     re_uid_map_file = dir_name + 'finally_re_uid_map_dict.pic'
#     re_aid_map_file = dir_name + 'finally_re_aid_map_dict.pic'
#
#     train_data, dev_data, predict_data, relevant_user_data, no_relevant_user_data, ad_data = pd.read_csv(train_data_file), \
#         pd.read_csv(dev_data_file), pd.read_csv(predict_data_file), \
#         pd.read_csv(relevant_user_data_file), pd.read_csv(no_relevant_user_data_file), pd.read_csv(ad_data_file)
#
#     feature_conf_dict = pickle.load(open(feature_conf_dict_file, 'r'))
#     re_uid_map = pickle.load(open(re_uid_map_file, 'r'))
#     re_aid_map = pickle.load(open(re_aid_map_file, 'r'))
#     return train_data, dev_data, predict_data, relevant_user_data, no_relevant_user_data, ad_data, feature_conf_dict, re_uid_map, re_aid_map

one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education',
                   'gender', 'house', 'os', 'ct', 'marriageStatus']
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
                  'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']


def pro_vec_fea(x):
    if type(x) != str or '|' not in x:
        return ''
    return x.split('|')[0]


def pro_one_user(tu, start_feature_conf_dict):
    global one_hot_feature, vector_feature
    col = []
    for fea in one_hot_feature:
        col.append(start_feature_conf_dict[fea] + int(tu[fea]))
    for fea in vector_feature:
        for i in pro_vec_fea(tu[fea]).split(' '):
            if len(i) == 0:
                continue
            col.append(start_feature_conf_dict[fea] + int(i))
    return col


def cluseter(formal, cl_num=None):
    if formal:
        dir_name = '../../DataSet/preliminary_contest_data/'

        if cl_num == 0:
            cluseter_num_name = ['cl1', 'cl2']#, 'cl7', 'cl8', 'cl9', 'cl10']
            cluseter_num_val = [200, 400]
        elif cl_num == 1:
            cluseter_num_name = ['cl3', 'cl4']  # , 'cl7', 'cl8', 'cl9', 'cl10']
            cluseter_num_val = [800, 1000]
    else:
        dir_name = '../../DataSet/small_preliminary_contest_data/'
        cluseter_num_name = ['cl1', 'cl2']
        cluseter_num_val = [10, 20]
    print cluseter_num_name, cluseter_num_val
    relevant_user_data_file = dir_name + 'finally_processed_data_user_relevant.csv'
    no_relevant_user_data_file = dir_name + 'finally_processed_data_user_no_rel.csv'
    feature_conf_dict_file = dir_name + 'finally_feature_conf_dict.pic'

    uid_list = []
    all_feature_file = dir_name + 'feature_coo_matrix.npz'
    uid_list_file = dir_name + 'kmeans_uid_list.pic'
    print 'read data start !'
    if not os.path.exists(all_feature_file):
        user_data = pd.concat([pd.read_csv(relevant_user_data_file), pd.read_csv(no_relevant_user_data_file)])
        feature_conf_dict = pickle.load(open(feature_conf_dict_file, 'r'))
        start_feature_conf_dict, ss = {}, 0
        for fea in one_hot_feature:
            start_feature_conf_dict[fea] = ss
            ss += feature_conf_dict[fea]
        for fea in vector_feature:
            start_feature_conf_dict[fea] = ss
            ss += feature_conf_dict[fea][0]

        all_col_list, all_row_list = [], []
        len_user_data = len(user_data)
        for i in tqdm(range(len_user_data)):
            tu = user_data.iloc[i]
            uid_list.append(tu['uid'])
            col_list = pro_one_user(tu, start_feature_conf_dict)
            all_col_list.extend(col_list)
            all_row_list.extend([i] * len(col_list))
        all_feature = coo_matrix(([1.0] * len(all_col_list), (all_row_list, all_col_list)), shape=(len_user_data, ss))
        sparse.save_npz(all_feature_file, all_feature)
        pickle.dump(uid_list, open(uid_list_file, 'w'))
        del user_data, all_row_list, all_col_list
        gc.collect()
    else:
        all_feature = sparse.load_npz(all_feature_file)
        uid_list = pickle.load(open(uid_list_file, 'r'))


    # del user_data; gc.collect()
    # print 'row, col', len_user_data, ss

    for i in range(len(cluseter_num_val)):
        print 'start: ', datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), cluseter_num_name[i], cluseter_num_val[i]
        clf = KMeans(
            # init='k-means++',
            init='random',
            n_clusters=cluseter_num_val[i],
            n_init=1,
            # precompute_distances=True,
            max_iter=30,
            tol=1e-4,
            n_jobs=4,
            verbose=1,
            copy_x=False,
        )
        y_pred = clf.fit_predict(all_feature)
        print 'end--: ', datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), cluseter_num_name[i], cluseter_num_val[i]
        d_k_map = dict(zip(uid_list, y_pred))
        user_data = pd.concat([pd.read_csv(relevant_user_data_file), pd.read_csv(no_relevant_user_data_file)])
        user_data[cluseter_num_name[i]] = user_data['uid'].apply(lambda x: d_k_map[x])
        user_data.to_csv(dir_name+'finally_processed_data_all_user_clusters_'+'_'.join(cluseter_num_name[:i+1])+'.csv', index=False)
        del user_data
        gc.collect()


if __name__ == '__main__':
    # cluseter(False)
    cluseter(True, cl_num=0)
    # cluseter(True, cl_num=1)
    pass


