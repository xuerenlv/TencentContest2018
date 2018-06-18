# coding:utf-8

import argparse
from sklearn.utils import shuffle
import gc
import os
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
from datetime import datetime

from read_data import get_prod_dataset

one_hot_feature_user = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'marriageStatus']
one_hot_feature_ad = ['advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']


def pro_vec_fea(x):
    if type(x) != str or '|' not in x:
        return ''
    return x.split('|')[0]


def pro_one_user(tu, start_feature_conf_dict):
    global one_hot_feature_user, vector_feature
    col = []
    for fea in one_hot_feature_user:
        if int(tu[fea]) != 0:  # 缺失值
            col.append(str(start_feature_conf_dict[fea] + int(tu[fea])))
    for fea in vector_feature:
        for i in pro_vec_fea(tu[fea]).split(' '):
            if len(i) == 0:
                continue
            col.append(str(start_feature_conf_dict[fea] + int(i)))
    return col


def pro_one_ad(ad, start_feature_conf_dict):
    global one_hot_feature_ad
    col = []
    for fea in one_hot_feature_ad:
        if int(ad[fea]) != 0:  # 缺失值
            col.append(str(start_feature_conf_dict[fea] + int(ad[fea])))
    # print len(col)
    return col


def trainlgb(graph_hyper_params):
    def construct_train_data(pos_train_data, neg_train_data, graph_hyper_params):
        # global pos_train_data, neg_train_data, start_neg
        pos_len, neg_len = len(pos_train_data), len(neg_train_data)
        # print start_neg, pos_len, neg_len
        if graph_hyper_params['neg_start'] * pos_len + graph_hyper_params['neg_size'] * pos_len < neg_len:
            this_neg_train_data = neg_train_data[graph_hyper_params['neg_start'] * pos_len: \
                                                 graph_hyper_params['neg_start'] * pos_len + graph_hyper_params[
                                                     'neg_size'] * pos_len]
        else:
            print 'fianl ! fianl ! fianl ! fianl !'
            this_neg_train_data = pd.concat([neg_train_data[graph_hyper_params['neg_start'] * pos_len:], neg_train_data[: pos_len - max(0,neg_len - graph_hyper_params['neg_start'] * pos_len)]])
        train_data = pd.concat([pos_train_data, this_neg_train_data])
        return shuffle(train_data)

    print 'read data start !'
    pos_train_data, neg_train_data, predict_data1, predict_data2, user_data, ad_data, feature_conf_dict, uid_map, aid_map = get_prod_dataset(
        graph_hyper_params['formal'])
    print 'read data done !'

    o_dev_size = graph_hyper_params['o_dev_size']
    dev_data = pd.concat([pos_train_data[:o_dev_size], neg_train_data[:o_dev_size]])
    pos_train_data, neg_train_data = pos_train_data[o_dev_size:], neg_train_data[o_dev_size:]
    print 'dev_size:', len(dev_data)
    print 'pos-neg-len:', len(pos_train_data), len(neg_train_data)

    train_data = construct_train_data(pos_train_data, neg_train_data, graph_hyper_params)
    if graph_hyper_params['only_train']:
        if graph_hyper_params['formal']:
            formal_set = set(list(train_data['uid']) + list(dev_data['uid']))
        else:
            formal_set = set(list(train_data['uid']) + list(dev_data['uid']) + [1, 2, 3, 4])
        user_data = user_data[user_data['uid'].isin(formal_set)]

    # print 'map row start'
    # uid_map_row, aid_map_row = dict(zip(user_data['uid'].values, np.arange(len(user_data)))), dict(zip(ad_data['aid'].values, np.arange(len(ad_data))))
    # print 'map row end'
    print feature_conf_dict
    start_feature_conf_dict, ss = {}, 0
    for fea in one_hot_feature_user:
        start_feature_conf_dict[fea] = ss
        ss += feature_conf_dict[fea]
    for fea in vector_feature:
        start_feature_conf_dict[fea] = ss
        ss += feature_conf_dict[fea][0]
    for fea in one_hot_feature_ad:
        start_feature_conf_dict[fea] = ss
        ss += feature_conf_dict[fea]
    print 'ss:', ss

    new_user_data = []
    for i in tqdm(range(len(user_data))):
        ut = user_data.iloc[i]
        user_col = pro_one_user(ut, start_feature_conf_dict)
        new_user_data.append({'uid': ut['uid'], 'user_sparse_col': user_col})
    del user_data
    gc.collect()
    new_user_data = pd.DataFrame(new_user_data)

    new_ad_data = []
    for i in tqdm(range(len(ad_data))):
        ad = ad_data.iloc[i]
        ad_col = pro_one_ad(ad, start_feature_conf_dict)
        new_ad_data.append({'aid': ad['aid'], 'ad_sparse_col': ad_col})
    new_ad_data = pd.DataFrame(new_ad_data)
    new_ad_data['creativeSize'] = ad_data['creativeSize']
    del ad_data
    gc.collect()

    train_data = pd.merge(train_data, new_user_data, on='uid', how='left')
    train_data = pd.merge(train_data, new_ad_data, on='aid', how='left')
    train_data['fea'] = train_data['user_sparse_col'].apply(lambda x: ' '.join(x)+' ') + train_data['ad_sparse_col'].apply(lambda x: ' '.join(x)+' ')

    dev_data = pd.merge(dev_data, new_user_data, on='uid', how='left')
    dev_data = pd.merge(dev_data, new_ad_data, on='aid', how='left')
    dev_data['fea'] = dev_data['user_sparse_col'].apply(lambda x: ' '.join(x) + ' ') + dev_data['ad_sparse_col'].apply(lambda x: ' '.join(x) + ' ')

    del new_user_data, new_ad_data
    gc.collect()

    train_data_X = train_data[['creativeSize']]
    dev_data_X = dev_data[['creativeSize']]

    cv = CountVectorizer()
    cv.fit(train_data['fea'])
    train_a = cv.transform(train_data['fea'])
    dev_a = cv.transform(dev_data['fea'])
    train_data_X = sparse.hstack((train_data_X, train_a))
    dev_data_X = sparse.hstack((dev_data_X, dev_a))

    train_y = train_data.pop('label')
    dev_y = dev_data.pop('label')

    print train_data_X.shape
    del train_data, dev_data
    gc.collect()

    import lightgbm as lgb
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=41, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=2000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.08, min_child_weight=50, random_state=2018, n_jobs=8
    )
    clf.fit(csr_matrix(train_data_X, dtype=np.float32), train_y, eval_set=[(csr_matrix(dev_data_X, dtype=np.float32), dev_y)], eval_metric='auc', early_stopping_rounds=300)

    return



def parse():
    args = argparse.ArgumentParser(description='Ten Con !')
    args.add_argument('--model', type=str, default='dmf', help='model type')
    args.add_argument('--opt', type=str, default='adam', help='opt')
    args.add_argument('--lr', type=float, default=0.0001, help='lr')
    args.add_argument('--ns', type=int, default=1, help='neg size')
    args.add_argument('--only_train', type=int, default=1, help='only train')
    args.add_argument('--shp', type=int, default=10, help='show peroid')
    args.add_argument('--neg_start', type=int, default=0, help='neg_start') # neg_start
    return args


if __name__ == '__main__':
    args = parse()
    args = args.parse_args()
    # graph_hyper_params = {'batch_size': 4, 'l2_reg_alpha':0.0, 'learn_rate': args.lr, 'show_peroid': 2,
    #                       'formal': False, 'epoch': 2, 'debug': True, 'o_dev_size': 2,
    #                       'creativeSize_pro': 'li_san', 'neg_size': 1, 'model': args.model, 'opt': args.opt,
    #                       'neg_start': args.neg_start, 'only_train': True}

    graph_hyper_params = {'batch_size': 128, 'l2_reg_alpha': 0.0, 'learn_rate': args.lr, 'show_peroid': args.shp,
                          'formal': True, 'epoch': 1, 'debug': False, 'o_dev_size': 10000,
                          'creativeSize_pro': 'li_san', 'neg_size': args.ns, 'model': args.model, 'opt': args.opt,
                          'neg_start': args.neg_start, 'only_train': True}

    trainlgb(graph_hyper_params)
    pass











