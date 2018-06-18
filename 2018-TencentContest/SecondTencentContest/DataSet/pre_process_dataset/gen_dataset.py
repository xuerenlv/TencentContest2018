# coding:utf-8

import pandas as pd
import argparse
from sklearn.utils import shuffle
import numpy as np
import gc
import os
import cPickle as pickle

user_feature_file = 'userFeature.data'
ad_feature_file = 'adFeature.csv'
train_file = 'train.csv'
test_file = 'test1.csv'

small_dir = '../small_preliminary_contest_data/'
formal_dir = '../preliminary_contest_data/'


# data 是所有的数据
feature_conf_dict = {}
one_hot_feature = ['aid', 'uid',
                   'LBS', 'age', 'carrier', 'consumptionAbility', 'education',
                   'gender', 'house', 'os', 'ct', 'marriageStatus', 'advertiserId',
                   'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']
vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
                  'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
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
        if x == -100:
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
        if x == -100:
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
            # print 'hh',x
            self.size = max(self.size, len(set(x.split(' '))))


uid_map_obj = ShrinkSep()
aid_map_obj = ShrinkSep()
# aid_map = {}
# uid_map = {}

def process_data(data):
    print 'process data !'
    print '\t process data sep'
    global feature_conf_dict
    for cl in one_hot_feature:
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
    for cl in vector_feature:
        if cl in data.columns.values.tolist():
            sl = GetShrinkSucc()
            data[cl].apply(sl)

            sh = ShrinkSucc(sl.size)
            data[cl] = data[cl].apply(sh)

            # print cl, sh.d, sl.size
            feature_conf_dict[cl] = (len(sh.d) + 1, sh.ms)
    print '\t process data done'
    return data


def split_data(data, dev_size):
    print 'split data !'

    # 训练集
    train = data[data.label != -1]
    predict = data[data.label == -1]

    # user_data= data[user_features]
    # ad_data= data[ad_features]
    del data
    gc.collect()

    pos_train = shuffle(train[train.label == 1])
    neg_train = shuffle(train[train.label == 0])

    # print len(pos_train), len(neg_train)
    pos_dev = pos_train.iloc[:dev_size/2]
    neg_dev = neg_train.iloc[:dev_size/2]
    pos_train = pos_train.iloc[dev_size/2:]
    neg_train = neg_train.iloc[dev_size/2:]
    gc.collect()
    train_data = pd.concat([pos_train, neg_train])
    dev_data = pd.concat([pos_dev, neg_dev])

    # train_y = train_data.pop('label').values
    # dev_y = dev_data.pop('label').values
    # print '去重前 u a：', len(user_data), len(ad_data)
    # user_data = user_data.drop_duplicates()
    # ad_data = ad_data.drop_duplicates()
    # print '去重后 u a：', len(user_data), len(ad_data)

    return train_data, dev_data, predict #, user_data, ad_data


def read_process_data(dir_name):
    global user_feature_file, ad_feature_file, train_file, test_file, one_hot_feature
    user_feature_file = dir_name + user_feature_file
    ad_feature_file = dir_name + ad_feature_file
    train_file = dir_name + train_file
    test_file = dir_name + test_file

    print 'read data !'
    userFeature_data = []
    with open(user_feature_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                if each_list[0] in one_hot_feature:
                    userFeature_dict[each_list[0]] = int(each_list[1])
                else:
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 500000 == 0:
                print i
    user_feature = pd.DataFrame(userFeature_data)
    gc.collect()
    user_feature = user_feature.fillna(-100)
    user_data= process_data(user_feature)


    # ad_feature
    ad_feature = pd.read_csv(ad_feature_file)
    ad_feature = ad_feature.fillna(-100)
    ad_data= process_data(ad_feature)

    # train and predict
    train = pd.read_csv(train_file, dtype = {'aid' : int, 'uid' : int, 'label' : int})
    predict = pd.read_csv(test_file)

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])

    print 'merge data !'
    # data = pd.merge(data, ad_feature, on='aid', how='left')
    # data = pd.merge(data, user_feature, on='uid', how='left')
    # data = data.fillna(-100)

    # print data['interest1'].values
    # print data[data['uid'] == 53811011]['house']
    # print user_feature[user_feature['uid'] == 53811011]['house']
    return process_data(data), user_data, ad_data


def get_dataset(dir_name, dev_size):
    global feature_conf_dict
    train_data_file = dir_name + 'finally_processed_data_train.csv'
    dev_data_file = dir_name + 'finally_processed_data_dev.csv'
    predict_data_file = dir_name + 'finally_processed_data_predict.csv'
    relevant_user_data_file = dir_name + 'finally_processed_data_user_relevant.csv'
    no_relevant_user_data_file = dir_name + 'finally_processed_data_user_no_rel.csv'
    ad_data_file = dir_name + 'finally_processed_data_ad.csv'
    feature_conf_dict_file = dir_name + 'finally_feature_conf_dict.pic'
    re_uid_map_file = dir_name + 'finally_re_uid_map_dict.pic'
    re_aid_map_file = dir_name + 'finally_re_aid_map_dict.pic'

    if os.path.exists(train_data_file):
        train_data, dev_data, predict_data, relevant_user_data, no_relevant_user_data, ad_data = pd.read_csv(train_data_file), \
                pd.read_csv(dev_data_file), pd.read_csv(predict_data_file), \
                pd.read_csv(relevant_user_data_file), pd.read_csv(no_relevant_user_data_file), pd.read_csv(ad_data_file)

        feature_conf_dict = pickle.load(open(feature_conf_dict_file, 'r'))
        re_uid_map = pickle.load(open(re_uid_map_file, 'r'))
        re_aid_map = pickle.load(open(re_aid_map_file, 'r'))
    else:
        data, user_data, ad_data = read_process_data(dir_name)
        train_data, dev_data, predict_data = split_data(data, dev_size)
        aid_map, uid_map = aid_map_obj.d, uid_map_obj.d
        re_uid_map = dict(zip(uid_map.values(), uid_map.keys()))
        re_aid_map = dict(zip(aid_map.values(), aid_map.keys()))
        # predict_data['ori_aid'] = predict_data['aid'].apply(lambda x: re_aid_map[x])
        # predict_data['ori_uid'] = predict_data['uid'].apply(lambda x: re_uid_map[x])

        feature_conf_dict['uid'] = len(uid_map) + 1
        feature_conf_dict['aid'] = len(aid_map) + 1

        # 过滤 uid
        source_data = pd.concat([train_data, dev_data, predict_data])
        relevant_user_data = user_data[user_data['uid'].isin(source_data['uid'].unique())]
        no_relevant_user_data = user_data[user_data['uid'].isin(set(user_data['uid'].unique()) - set(source_data['uid'].unique()))]
        del user_data
        gc.collect()

        train_data.to_csv(train_data_file, index=False)
        dev_data.to_csv(dev_data_file, index=False)
        predict_data.to_csv(predict_data_file, index=False)
        relevant_user_data.to_csv(relevant_user_data_file, index=False)
        no_relevant_user_data.to_csv(no_relevant_user_data_file, index=False)
        ad_data.to_csv(ad_data_file, index=False)
        pickle.dump(feature_conf_dict, open(feature_conf_dict_file, 'w'))
        pickle.dump(re_uid_map, open(re_uid_map_file, 'w'))
        pickle.dump(re_aid_map, open(re_aid_map_file, 'w'))

    # print train_data
    print feature_conf_dict
    return train_data, dev_data, predict_data, relevant_user_data, no_relevant_user_data, ad_data, feature_conf_dict, re_uid_map, re_aid_map


def parse():
    args = argparse.ArgumentParser(description='Ten Con !')
    args.add_argument('--fm', type=bool, default=False, help='formal')
    return args


if __name__ == '__main__':
    args = parse()
    args = args.parse_args()
    if args.fm:
        get_dataset(formal_dir, dev_size=4000)
    else:
        get_dataset(small_dir, dev_size=10)
