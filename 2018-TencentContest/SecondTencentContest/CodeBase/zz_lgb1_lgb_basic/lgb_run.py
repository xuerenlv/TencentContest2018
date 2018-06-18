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

from read_data import get_prod_dataset, save_file, get_user_data

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
        if 'LBS' not in fea:
            if int(tu[fea]) != 0:  # 缺失值
                col.append(str(start_feature_conf_dict[fea] + int(tu[fea])))
        else:
            if int(tu[fea]) != 0:  # 缺失值
                col.append(fea + str(start_feature_conf_dict[fea] + int(tu[fea])))
    for fea in vector_feature:
        if 'kw' not in fea and 'appId' not in fea and 'topic' not in fea:
            for i in pro_vec_fea(tu[fea]).split(' '):
                if len(i) == 0:
                    continue
                col.append(str(start_feature_conf_dict[fea] + int(i)))
        else:
            # print fea
            for i in pro_vec_fea(tu[fea]).split(' '):
                if len(i) == 0:
                    continue
                col.append(fea + str(start_feature_conf_dict[fea] + int(i)))
    return col


def pro_one_ad(ad, start_feature_conf_dict):
    global one_hot_feature_ad
    col = []
    for fea in one_hot_feature_ad:
        if 'creativeId' not in fea :
            if int(ad[fea]) != 0:  # 缺失值
                col.append(str(start_feature_conf_dict[fea] + int(ad[fea])))
        else:
            if int(ad[fea]) != 0:  # 缺失值
                col.append(fea + str(start_feature_conf_dict[fea] + int(ad[fea])))
    # print len(col)
    return col


def cross_fea(uss, add):
    res = []
    for i in tqdm(range(len(uss))):
        tt = []
        uu, aa = uss[i], add[i]
        for j in range(len(aa)):
            for t in range(len(uu)):
                if 'creativeId' in aa[j]:
                    continue
                if 'appId' in uu[t] or 'kw' in uu[t] or 'topic' in uu[t] or 'LBS' in uu[t]:
                    continue
                tt.append(aa[j] + uu[t])
        res.append(tt)
    return res


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
    dev_data = pd.merge(dev_data, new_user_data, on='uid', how='left')
    dev_data = pd.merge(dev_data, new_ad_data, on='aid', how='left')
    del new_user_data
    gc.collect()

    train_data['cross'] = cross_fea(train_data['user_sparse_col'], train_data['ad_sparse_col'])
    train_data['fea'] = train_data['user_sparse_col'].apply(lambda x: ' '.join(x) + ' ') + train_data[
        'ad_sparse_col'].apply(lambda x: ' '.join(x) + ' ') + train_data['cross'].apply(lambda x: ' '.join(x) + ' ')
    train_data_X = train_data[['creativeSize']]
    cv = CountVectorizer()
    cv.fit(train_data['fea'])
    train_a = cv.transform(train_data['fea'])
    train_data_X = sparse.hstack((train_data_X, train_a))
    train_y = train_data['label']
    print train_data_X.shape
    train_data_ori = train_data[['uid', 'aid', 'label']]
    del train_data
    gc.collect()

    dev_data['cross'] = cross_fea(dev_data['user_sparse_col'], dev_data['ad_sparse_col'])
    dev_data['fea'] = dev_data['user_sparse_col'].apply(lambda x: ' '.join(x) + ' ') + dev_data['ad_sparse_col'].apply(
        lambda x: ' '.join(x) + ' ') + dev_data['cross'].apply(lambda x: ' '.join(x) + ' ')
    dev_data_X = dev_data[['creativeSize']]
    dev_a = cv.transform(dev_data['fea'])
    dev_data_X = sparse.hstack((dev_data_X, dev_a))
    dev_y = dev_data['label']
    dev_data_ori = dev_data[['uid', 'aid', 'label']]
    del dev_data
    gc.collect()


    import lightgbm as lgb
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=41, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1500, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.08, min_child_weight=50, random_state=2018, n_jobs=10
    )
    clf.fit(csr_matrix(train_data_X, dtype=np.float32), train_y, eval_set=[(csr_matrix(dev_data_X, dtype=np.float32), dev_y)], eval_metric='auc', early_stopping_rounds=100)

    train_data_ori['leaf'] = clf.predict(csr_matrix(train_data_X, dtype=np.float32), raw_score=True)
    dev_data_ori['leaf'] = clf.predict(csr_matrix(dev_data_X, dtype=np.float32), raw_score=True)
    del train_data_X, train_y, dev_data_X, dev_y
    gc.collect()

    predict_data = predict_data2.sort_values(by='uid')
    bat_size = len(predict_data)/10
    s = 0
    user_data = get_user_data(graph_hyper_params)
    bfinal_predict_datall, bpredict_data_orill = [], []
    for i in range(10):
        if i != 9:
            pp = predict_data.iloc[s:s+bat_size, :]
            s += bat_size
        else:
            pp = predict_data.iloc[s:, :]
        bfinal_predict_data, bpredict_data_ori = batch_predict(pp, start_feature_conf_dict, new_ad_data, clf, user_data, cv)
        bfinal_predict_datall.append(bfinal_predict_data)
        bpredict_data_orill.append(bpredict_data_ori)

    final_predict_data = pd.concat(bfinal_predict_datall)
    predict_data_ori = pd.concat(bpredict_data_orill)

    save_file(final_predict_data, predict_data_ori, train_data_ori, dev_data_ori, graph_hyper_params)
    return


def batch_predict(predict_data, start_feature_conf_dict, new_ad_data, clf, user_data, cv):
    formal_set = set(list(predict_data['uid']))
    user_d = user_data[user_data['uid'].isin(formal_set)]
    new_user_data = []
    for i in tqdm(range(len(user_d))):
        ut = user_d.iloc[i]
        user_col = pro_one_user(ut, start_feature_conf_dict)
        new_user_data.append({'uid': ut['uid'], 'user_sparse_col': user_col})
    new_user_data = pd.DataFrame(new_user_data)

    predict_data = pd.merge(predict_data, new_user_data, on='uid', how='left')
    predict_data = pd.merge(predict_data, new_ad_data, on='aid', how='left')
    predict_data['cross'] = cross_fea(predict_data['user_sparse_col'], predict_data['ad_sparse_col'])
    predict_data['fea'] = predict_data['user_sparse_col'].apply(lambda x: ' '.join(x) + ' ') + predict_data[
        'ad_sparse_col'].apply(lambda x: ' '.join(x) + ' ') + predict_data['cross'].apply(lambda x: ' '.join(x) + ' ')
    predict_data_X = predict_data[['creativeSize']]
    predict_a = cv.transform(predict_data['fea'])
    predict_data_X = sparse.hstack((predict_data_X, predict_a))
    predict_data_ori = predict_data[['uid', 'aid']]
    final_predict_data = predict_data[['uid', 'aid']]
    del predict_data
    gc.collect()

    final_predict_data['score'] = clf.predict_proba(csr_matrix(predict_data_X, dtype=np.float32))[:, 1]
    predict_data_ori['leaf'] = clf.predict(csr_matrix(predict_data_X, dtype=np.float32), raw_score=True)

    return final_predict_data, predict_data_ori



def parse():
    args = argparse.ArgumentParser(description='Ten Con !')
    args.add_argument('--ns', type=int, default=1, help='neg size')
    args.add_argument('--only_train', type=int, default=1, help='only train')
    args.add_argument('--neg_start', type=int, default=0, help='neg_start') # neg_start
    return args


if __name__ == '__main__':
    args = parse()
    args = args.parse_args()

    # graph_hyper_params = {'formal': False, 'o_dev_size': 2, 'neg_size': 1, 'neg_start': args.neg_start, 'only_train': True}
    graph_hyper_params = {'formal': True, 'o_dev_size': 10000, 'neg_size': args.ns, 'neg_start': args.neg_start, 'only_train': True}

    trainlgb(graph_hyper_params)
    pass











