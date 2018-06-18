# coding:utf-8

import pandas as pd
import argparse
from sklearn.utils import shuffle
import numpy as np
import gc
import os
import cPickle as pickle
from tqdm import tqdm

user_feature_file = 'userFeature.data'
ad_feature_file = 'adFeature.csv'
train_file = 'train.csv'
test_file1 = 'test1.csv'
test_file2 = 'test2.csv'

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
user_features = ['uid','LBS', 'age', 'carrier', 'consumptionAbility', 'education',
                 'gender', 'house', 'os', 'ct', 'marriageStatus',
                 'appIdAction', 'appIdInstall', 'interest1', 'interest2',
                 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
ad_features = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType', 'creativeSize']





class ShrinkSep:
    def __init__(self):
        self.d = {}

    def __call__(self, x):
        if x == -100 or x == 0:
            return 0
        if x not in self.d:
            self.d[x] = len(self.d) + 1
        return self.d[x]


class ShrinkSucc:
    def __init__(self, size):
        self.d = {}
        self.ms = size
        self.neg = str(self.ms)

    def __call__(self, x):
        if x == -100 or x == 0:
            return self.neg
        x = [int(i) for i in x.split(' ')]
        sx = set(x)
        index = []
        for i in sx:
            if i not in self.d:
                self.d[i] = len(self.d) + 1
            index.append(self.d[i])

        # print 'aa', index + [0]*(self.ms-len(sx)), val + [0.0]*(self.ms-len(sx))
        # 当前交互的项目 | 最长的交互项目长度 | 当前交互项目的长度
        return ' '.join([str(k) for k in index]) + '|' + self.neg + "|" + str(len(index))


class GetShrinkSucc:
    def __init__(self):
        self.size = 0

    def __call__(self, x):
        if x != -100:
            self.size = max(self.size, len(set(x.split(' '))))


uid_map_obj = ShrinkSep()
aid_map_obj = ShrinkSep()
def process_data(data):
    print 'process data !'
    print '\t process data sep'
    global feature_conf_dict
    for cl in tqdm(one_hot_feature):
        if cl == 'aid':
            sh = aid_map_obj
        elif cl == 'uid':
            sh = uid_map_obj
        else:
            sh = ShrinkSep()

        if cl in data.columns.values.tolist():
            data[cl] = data[cl].apply(sh)
            if cl != 'aid' and cl != 'uid':
                feature_conf_dict[cl] = len(sh.d) + 1

    print '\t process data suc'
    for cl in tqdm(vector_feature):
        if cl in data.columns.values.tolist():
            sl = GetShrinkSucc()
            data[cl].apply(sl)
            sh = ShrinkSucc(sl.size)
            data[cl] = data[cl].apply(sh)
            feature_conf_dict[cl] = (len(sh.d) + 1, sh.ms)
    print '\t process data done'
    return data


def read_process_data(dir_name):
    global user_feature_file, ad_feature_file, train_file, test_file1, test_file2, one_hot_feature
    user_feature_file = dir_name + user_feature_file
    ad_feature_file = dir_name + ad_feature_file
    train_file = dir_name + train_file
    test_file1 = dir_name + test_file1
    test_file2 = dir_name + test_file2

    print 'read data !'
    userFeature_data = []
    with open(user_feature_file, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                if each_list[0] in one_hot_feature:
                    userFeature_dict[each_list[0]] = int(each_list[1])
                else:
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
    print '1'
    user_feature = pd.DataFrame(userFeature_data)
    print '2'
    del userFeature_data
    gc.collect()
    print '3'
    user_feature = user_feature.fillna(-100)
    print '4'
    user_data = process_data(user_feature)

    # ad_feature
    ad_feature = pd.read_csv(ad_feature_file)
    ad_feature = ad_feature.fillna(-100)
    ad_data = process_data(ad_feature)

    # predict
    predict_data1 = pd.read_csv(test_file1, dtype={'aid' : int, 'uid' : int, 'label' : int})
    predict_data2 = pd.read_csv(test_file2, dtype={'aid' : int, 'uid' : int, 'label' : int})
    predict_data1['label'] = -1
    predict_data2['label'] = -1

    predict_data1['ori_aid'], predict_data1['ori_uid'] = predict_data1['aid'], predict_data1['uid']
    predict_data2['ori_aid'], predict_data2['ori_uid'] = predict_data2['aid'], predict_data2['uid']

    global uid_map_obj, aid_map_obj
    aid_map, uid_map = aid_map_obj.d, uid_map_obj.d
    predict_data1['aid'], predict_data1['uid'] = predict_data1['aid'].apply(lambda x: aid_map[x]), predict_data1['uid'].apply(lambda x: uid_map[x])
    predict_data2['aid'], predict_data2['uid'] = predict_data2['aid'].apply(lambda x: aid_map[x]), predict_data2['uid'].apply(lambda x: uid_map[x])

    # train
    train = pd.read_csv(train_file, dtype={'aid': int, 'uid': int, 'label': int})
    train['aid'], train['uid'] = train['aid'].apply(lambda x: aid_map[x]), train['uid'].apply(lambda x: uid_map[x])
    train.loc[train['label'] == -1, 'label'] = 0
    pos_train = train[train['label'] == 1]
    neg_train = train[train['label'] == 0]
    print 'pos_train_len', 'neg_train_len', len(pos_train), len(neg_train)

    # feature_conf_dict
    feature_conf_dict['uid'] = len(uid_map) + 1
    feature_conf_dict['aid'] = len(aid_map) + 1

    return user_data, ad_data, pos_train, neg_train, predict_data1, predict_data2, aid_map, uid_map


def get_dataset(dir_name):
    global feature_conf_dict
    pos_train_data_file = dir_name + 'finally_processed_data_train_pos.csv'
    neg_train_data_file = dir_name + 'finally_processed_data_train_neg.csv'
    predict_data_file1 = dir_name + 'finally_processed_data_predict_1.csv'
    predict_data_file2 = dir_name + 'finally_processed_data_predict_2.csv'
    user_data_file = dir_name + 'finally_processed_data_user.csv'
    ad_data_file = dir_name + 'finally_processed_data_ad.csv'
    feature_conf_dict_file = dir_name + 'finally_feature_conf_dict.pic'

    uid_map_file = dir_name + 'finally_uid_map_dict.pic'
    aid_map_file = dir_name + 'finally_aid_map_dict.pic'

    user_data, ad_data, pos_train, neg_train, predict_data1, predict_data2, aid_map, uid_map = read_process_data(dir_name)
    user_data.to_csv(user_data_file, index=False)
    ad_data.to_csv(ad_data_file, index=False)
    pos_train.to_csv(pos_train_data_file, index=False)
    neg_train.to_csv(neg_train_data_file, index=False)
    predict_data1.to_csv(predict_data_file1, index=False)
    predict_data2.to_csv(predict_data_file2, index=False)
    pickle.dump(feature_conf_dict, open(feature_conf_dict_file, 'w'))

    pickle.dump(uid_map, open(uid_map_file, 'w'))
    pickle.dump(aid_map, open(aid_map_file, 'w'))

    print feature_conf_dict



def parse():
    args = argparse.ArgumentParser(description='Ten Con !')
    args.add_argument('--fm', type=bool, default=False, help='formal')
    return args


if __name__ == '__main__':
    args = parse()
    args = args.parse_args()
    if args.fm:
        get_dataset(formal_dir)
    else:
        get_dataset(small_dir)
