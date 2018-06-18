# coding:utf-8

import pandas as pd
import argparse
from sklearn.utils import shuffle
import numpy as np

user_feature_file = 'userFeature.data'
ad_feature_file = 'adFeature.csv'
train_file = 'train.csv'
test_file = 'test1.csv'

small_dir = '../small_preliminary_contest_data/'
formal_dir = '../preliminary_contest_data/'


def analysis_train_file(dir_name, dev_size):
    global train_file
    train_file = dir_name + train_file
    print 'analysis:', train_file
    train_data = pd.read_csv(train_file)

    print '交互数目：', len(train_data)
    print '用户数：', len(set(train_data['uid']))
    print '广告数：', len(set(train_data['aid']))
    print 'label数: ', len(set(train_data['label']))
    print 'label，正例: ', len(train_data[train_data['label'] > 0])
    print 'label，负例: ', len(train_data[train_data['label'] < 0])

    # 生成 train - dev
    formal_train_file_positive = dir_name + 'formal_train_file_positive.csv'
    formal_train_file_negtative = dir_name + 'formal_train_file_negtative.csv'
    formal_dev_file = dir_name + 'formal_dev_file.csv'

    pos_data = shuffle(train_data[train_data['label'] > 0])
    neg_data = shuffle(train_data[train_data['label'] < 0])


    with open(formal_dev_file, 'w') as fw:
        for i in range(dev_size):
            pda, nda = pos_data.iloc[i], neg_data.iloc[i]
            fw.write(str(pda['aid']) + ' ' + str(pda['uid']) + ' 1' + '\n')
            fw.write(str(nda['aid']) + ' ' + str(nda['uid']) + ' 0' + '\n')

    with open(formal_train_file_positive, 'w') as fw:
        for i in range(dev_size, len(pos_data)):
            pda = pos_data.iloc[i]
            fw.write(str(pda['aid']) + ' ' + str(pda['uid']) + ' 1' + '\n')

    with open(formal_train_file_negtative, 'w') as fw:
        for i in range(dev_size, len(neg_data)):
            nda = neg_data.iloc[i]
            fw.write(str(nda['aid']) + ' ' + str(nda['uid']) + ' 0' + '\n')


def analysis_ad_feature_file(dir_name):
    global ad_feature_file
    ad_feature_file = dir_name + ad_feature_file
    print 'analysis:', ad_feature_file
    ad_data = pd.read_csv(ad_feature_file)
    ad_data = ad_data.apply(pd.to_numeric)

    cols = [u'aid', u'advertiserId', u'campaignId', u'creativeId',
            u'creativeSize', u'adCategoryId', u'productId',
            u'productType']
    for co in cols:
        if not co == 'creativeSize' and not co == 'aid':
            da = ad_data[co].unique()
            da_map = pd.Series(data=np.arange(len(da)), index=da)
            ad_data = pd.merge(ad_data, pd.DataFrame({co: da, co + '_pre': da_map[da].values}),
                               on=co, how='inner')

    # print ad_data
    prod_ad_file = dir_name + 'formal_adFeature.txt'
    with open(prod_ad_file, 'w') as fw:
        for i in range(len(ad_data)):
            ada = ad_data.iloc[i]
            r_str = '{} {}'.format(ada['aid'], ada['creativeSize'])
            for co in cols:
                if not co == 'creativeSize' and not co == 'aid':
                    r_str += ' ' + str(ada[co + '_pre'])
            fw.write(r_str + "\n")


def pro_feature_group(fg):
    fg_li = fg.split(' ')
    fgn = fg_li[0]
    fgn_li = [int(i) for i in fg_li[1:]]
    return fgn, fgn_li



def analysis_user_feature_file(dir_name):
    global user_feature_file
    user_feature_file = dir_name + user_feature_file
    print 'analysis:', user_feature_file

    userFeature_data = []
    with open(user_feature_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        # user_feature.to_csv('../data/userFeature.csv', index=False)
    print user_feature



    # print ad_data
    # prod_user_file = dir_name + 'formal_userFeature.txt'
    # with open(prod_user_file, 'w') as fw:

    pass


def parse():
    args = argparse.ArgumentParser(description='Analysis Data !')
    args.add_argument('--fm', type=bool, default=False, help='sum the integers (default: find the max)')
    return args


if __name__ == '__main__':
    args = parse()
    args = args.parse_args()

    if args.fm:
        # analysis_train_file(formal_dir, 2000)
        # analysis_ad_feature_file(formal_dir)
        analysis_user_feature_file(formal_dir)
    else:
        # analysis_train_file(small_dir, 5)
        # analysis_ad_feature_file(small_dir)
        analysis_user_feature_file(small_dir)

    pass
