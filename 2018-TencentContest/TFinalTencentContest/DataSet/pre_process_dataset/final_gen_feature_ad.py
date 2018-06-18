# coding:utf-8

import pandas as pd
import argparse
from sklearn.utils import shuffle
import numpy as np
import gc
import os
import cPickle as pickle
from tqdm import tqdm
import functools



small_dir = '../small_preliminary_contest_data/'
formal_dir = '../preliminary_contest_data/'


# data 是所有的数据
feature_conf_dict = {}
one_hot_feature = ['aid', 'uid',
                   'LBS', 'age', 'carrier', 'consumptionAbility', 'education',
                   'gender', 'house', 'os', 'ct', 'advertiserId',
                   'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
                  'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'marriageStatus']
continuous_feature = 'creativeSize'

# 用户和广告的特征
# user_features = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education',
#                  'gender', 'house', 'os', 'ct', 'marriageStatus',
#                  'appIdAction', 'appIdInstall', 'interest1', 'interest2',
#                  'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

prod_user_features = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct']

ad_features = ['advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType', 'creativeSize']





def pro_dataset(dir_name):
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
        pd.read_csv(neg_train_data_file), pd.read_csv(predict_data_file1), pd.read_csv(predict_data_file2), \
        pd.read_csv(user_data_file), pd.read_csv(ad_data_file)

    feature_conf_dict = pickle.load(open(feature_conf_dict_file, 'r'))
    uid_map = pickle.load(open(uid_map_file, 'r'))
    aid_map = pickle.load(open(aid_map_file, 'r'))

    print 'start-1'
    print prod_user_features
    prod_user_features_counts = {}
    ss = 0
    for fea in prod_user_features:
        prod_user_features_counts[fea] = {}
        prod_user_features_counts[fea]['dif'] = user_data[fea].unique()
        # prod_user_features_counts[fea]['counts'] = user_data[fea].value_counts()
        print fea, len(prod_user_features_counts[fea]['dif'])
        ss += len(prod_user_features_counts[fea]['dif'])


    all_list = np.array([1.0] * ss)
    def adfa(aid, all_list):
        pos_users = user_data[user_data['uid'].isin(pos_train_data[pos_train_data['aid'] == aid]['uid'])]
        ress = []
        for fea in prod_user_features:
            posc = pos_users[fea].value_counts()
            fea_set = prod_user_features_counts[fea]['dif']
            res = [0] * len(fea_set)
            for i in range(len(fea_set)):
                res[i] = (0 if fea_set[i] not in posc else posc[fea_set[i]])*1.0 #/prod_user_features_counts[fea]['counts'][fea_set[i]]
            ress.extend(res)
        all_list += np.array(ress)
        print aid, len(ress)
        return ress

    print 'start-2'
    ad_data['prod_fea'] = ad_data['aid'].apply(functools.partial(adfa, all_list=all_list))
    print all_list
    print 'start-3'

    ad_data['prod_fea'] = ad_data['prod_fea'].apply(lambda x: list(x/all_list))
    print ad_data.head()

    ad_data_file = dir_name + 'finally_processed_data_feature_engine_ad.csv'
    ad_data.to_csv(ad_data_file, index=False)



def parse():
    args = argparse.ArgumentParser(description='Ten Con !')
    args.add_argument('--fm', type=bool, default=False, help='formal')
    return args


if __name__ == '__main__':
    args = parse()
    args = args.parse_args()
    if args.fm:
        pro_dataset(formal_dir)
    else:
        pro_dataset(small_dir)
