# coding:utf-8

import pandas as pd
import argparse
from sklearn.utils import shuffle
import numpy as np
import gc
import os
import cPickle as pickle
import tensorflow as tf
from datetime import datetime
import copy

from metrics import gini_norm
from models import inference
# from models_croos_field_fm import inference
from read_data import get_prod_dataset
from sklearn import metrics
from tqdm import tqdm



user_features = ['uid', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education',
                 'gender', 'house', 'os', 'ct', 'marriageStatus',
                 'appIdAction', 'appIdInstall', 'interest1', 'interest2',
                 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
ad_features = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType',
               'creativeSize']

ad_features_for_cross = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType',
               'creativeSize_cross']

class SplitClass:
    def __init__(self):
        self.split_cache, self.split_cache_rem_size = {}, {}
        self.split_cache_interest, self.split_cache_rem_size_interest = {}, {}

    def clean(self):
        d_key = []
        barrier1 = max(self.split_cache_rem_size.values()) * 9 / 10
        for k in self.split_cache_rem_size:
            if self.split_cache_rem_size[k] <= barrier1:
                d_key.append(k)
            else:
                self.split_cache_rem_size[k] = 0
        for k in d_key:
            del self.split_cache[k], self.split_cache_rem_size[k]


        d_key = []
        barrier2 = max(self.split_cache_rem_size_interest.values()) * 9 / 10
        for k in self.split_cache_rem_size_interest:
            if self.split_cache_rem_size_interest[k] <= barrier2:
                d_key.append(k)
            else:
                self.split_cache_rem_size_interest[k] = 0
        for k in d_key:
            del self.split_cache_interest[k], self.split_cache_rem_size_interest[k]

        gc.collect()
        pass

    def __call__(self, vda, interest=None, feature_config=None):
        index_data, val_data = [], [] #, , len_data = [], [], []

        if interest is not None:
            for d in vda:
                if type(d) != str or '|' not in d:
                    if interest not in self.split_cache_interest:
                        ind, val = [0] * feature_config[interest][0], [0.0] * feature_config[interest][0]
                        index_data.append(ind)
                        val_data.append(val)
                        self.split_cache_interest[interest] = (ind, val)
                        self.split_cache_rem_size_interest[interest] = 1
                    else:
                        index_data.append(self.split_cache_interest[interest][0])
                        val_data.append(self.split_cache_interest[interest][1])
                        self.split_cache_rem_size_interest[interest] += 1
                else:
                    if d not in self.split_cache_interest:
                        ind, val = [0] * feature_config[interest][0], [0.0] * feature_config[interest][0]
                        dl = d.split('|')
                        for ori_ind in dl[0].split(' '):
                            ind[int(ori_ind)] = int(ori_ind)
                            val[int(ori_ind)] = 1.0
                        index_data.append(ind)
                        val_data.append(val)
                        self.split_cache_interest[d] = (ind, val)
                        self.split_cache_rem_size_interest[d] = 1
                    else:
                        index_data.append(self.split_cache_interest[d][0])
                        val_data.append(self.split_cache_interest[d][1])
                        self.split_cache_rem_size_interest[d] += 1
            return np.array(index_data).astype(np.int32), np.array(np.expand_dims(val_data, axis=1)).astype(np.float32), None


        # print 1211
        # print 1212
        for d in vda:
            # print 'aa',d, 'cc', type(d),'bb'
            if type(d) != str or '|' not in d:
                mx_len = d
                if mx_len not in self.split_cache:
                    # print 1213
                    mx_len = int(d)
                    ind, val = [0] * mx_len, [0.0] * mx_len
                    index_data.append(ind)
                    val_data.append(val)
                    self.split_cache[mx_len] = (ind, val)
                    self.split_cache_rem_size[mx_len] = 0
                    # print 1213
                else:
                    # print 1214
                    index_data.append(self.split_cache[mx_len][0])
                    val_data.append(self.split_cache[mx_len][1])
                    self.split_cache_rem_size[mx_len] += 1
                    # print 1214
            else:
                if d not in self.split_cache:
                    # print 1215
                    dl = d.split('|')
                    # len_data.append(int(dl[2]))
                    mx_len = int(dl[1])
                    t_len = int(dl[2])
                    ind, val = dl[0].split(' ') + [0] * (mx_len-t_len), [1.0]*t_len + [0.0] * (mx_len-t_len)
                    index_data.append(ind)
                    val_data.append(val)
                    self.split_cache[d] = (ind, val)
                    self.split_cache_rem_size[d] = 0
                    # print 1215
                else:
                    # print 1216
                    index_data.append(self.split_cache[d][0])
                    val_data.append(self.split_cache[d][1])
                    self.split_cache_rem_size[d] += 1
                    # print 1216
        # return np.array(index_data).astype(np.int32), np.array(np.expand_dims(val_data, axis=1)).astype(np.float32), np.array(len_data).astype(np.int32)
        return np.array(index_data).astype(np.int32), np.array(np.expand_dims(val_data, axis=1)).astype(np.float32), None

class ShrinkSep:
    def __init__(self):
        self.d = {}

    def __call__(self, x):
        if x == -100 or x==0:
            return 0
        if x not in self.d:
            self.d[x] = len(self.d) + 1
        return self.d[x]

def train_eval_model(graph_hyper_params):
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

    print graph_hyper_params

    print 'read data start !'
    pos_train_data, neg_train_data, predict_data1, predict_data2, user_data, ad_data, feature_conf_dict, uid_map, aid_map = get_prod_dataset(graph_hyper_params['formal'])
    print 'read data done !'

    # 重新 split train dev
    # o_dev_size = graph_hyper_params['o_dev_size']
    # dev_data = pd.concat([pos_train_data[:o_dev_size], neg_train_data[:o_dev_size]])
    # pos_train_data, neg_train_data = pos_train_data[o_dev_size:], neg_train_data[o_dev_size:]
    # print 'dev_size:', len(dev_data)
    # print 'pos-neg-len:', len(pos_train_data), len(neg_train_data)

    train_data = construct_train_data(pos_train_data, neg_train_data, graph_hyper_params)
    # if graph_hyper_params['only_train']:
    #     if graph_hyper_params['formal']:
    #         formal_set = set(list(train_data['uid']) + list(dev_data['uid']))
    #     else:
    #         formal_set = set(list(train_data['uid']) + list(dev_data['uid']) + [1, 2, 3, 4])
    #     user_data = user_data[user_data['uid'].isin(formal_set)]

    print 'map row start'
    user_data_train = user_data[user_data['uid'].isin(train_data['uid'])]
    user_data_predict1 = user_data[user_data['uid'].isin(predict_data1['uid'])]
    user_data_predict2 = user_data[user_data['uid'].isin(predict_data2['uid'])]
    del user_data
    gc.collect()
    uid_map_row_train, aid_map_row = dict(zip(user_data_train['uid'].values, np.arange(len(user_data_train)))), dict(zip(ad_data['aid'].values, np.arange(len(ad_data))))
    uid_map_row_predict_1 = dict(zip(user_data_predict1['uid'].values, np.arange(len(user_data_predict1))))
    uid_map_row_predict_2 = dict(zip(user_data_predict2['uid'].values, np.arange(len(user_data_predict2))))
    print 'map row end'
    print feature_conf_dict


    # graph = tf.Graph()
    # with graph.as_default():
        # 对 creativeSize 这一个连续特征的处理
    if graph_hyper_params['creativeSize_pro'] == 'min_max':
        print 'min-max norm creativeSize', ad_data['creativeSize'].max(), ad_data['creativeSize'].min()
        norm_cs = (ad_data['creativeSize'] * 1.0 - ad_data['creativeSize'].min()) / (
                ad_data['creativeSize'].max() - ad_data['creativeSize'].min())
        ad_data = ad_data.drop(['creativeSize'], axis=1)
        ad_data['creativeSize'] = norm_cs
        creativesize_p = tf.placeholder(tf.float32, [None, 1], name="creativeSize")
    elif graph_hyper_params['creativeSize_pro'] == 'li_san':
        print '离散化 creativeSize'
        sh = ShrinkSep()
        ad_data['creativeSize'] = ad_data['creativeSize'].apply(sh)
        feature_conf_dict['creativeSize'] = len(sh.d) + 1
        creativesize_p = tf.placeholder(tf.int32, [None, 1], name="creativeSize")
    else:
        print 'no process creativeSize'
    # ****************************************************************** place holder start
    uid_p = tf.placeholder(tf.int32, [None, 1], name="uid")
    lbs_p = tf.placeholder(tf.int32, [None, 1], name="LBS")
    age_p = tf.placeholder(tf.int32, [None, 1], name="age")

    carrier_p = tf.placeholder(tf.int32, [None, 1], name="carrier")
    consumptionability_p = tf.placeholder(tf.int32, [None, 1], name="consumptionAbility")
    education_p = tf.placeholder(tf.int32, [None, 1], name="education")
    gender_p = tf.placeholder(tf.int32, [None, 1], name="gender")
    house_p = tf.placeholder(tf.int32, [None, 1], name="house")
    os_p = tf.placeholder(tf.int32, [None, 1], name="os")
    ct_p = tf.placeholder(tf.int32, [None, 1], name="ct")
    # marriagestatus_p = tf.placeholder(tf.int32, [None, 1], name="marriageStatus")


    appidaction_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['appIdAction'][1]], name="appidaction_index")
    appidaction_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['appIdAction'][1]], name="appidaction_val")
    appIdInstall_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['appIdInstall'][1]], name="appIdInstall_index")
    appIdInstall_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['appIdInstall'][1]], name="appIdInstall_val")


    marriagestatus_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['marriageStatus'][0]], name="marriageStatus_index")
    marriagestatus_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['marriageStatus'][0]], name="marriageStatus_val")
    interest1_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['interest1'][0]], name="interest1_index")
    interest1_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['interest1'][0]], name="interest1_val")
    interest2_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['interest2'][0]], name="interest2_index")
    interest2_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['interest2'][0]], name="interest2_val")
    interest3_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['interest3'][0]], name="interest3_index")
    interest3_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['interest3'][0]], name="interest3_val")
    interest4_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['interest4'][0]], name="interest4_index")
    interest4_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['interest4'][0]], name="interest4_val")
    interest5_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['interest5'][0]], name="interest5_index")
    interest5_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['interest5'][0]], name="interest5_val")

    # kmeans type
    # clu_200_p = tf.placeholder(tf.int32, [None, 1], name="clu_200_p")
    # clu_400_p = tf.placeholder(tf.int32, [None, 1], name="clu_400_p")


    kw1_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['kw1'][1]], name="kw1_index")
    kw1_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['kw1'][1]], name="kw1_val")
    kw2_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['kw2'][1]], name="kw2_index")
    kw2_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['kw2'][1]], name="kw2_val")
    kw3_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['kw3'][1]], name="kw3_index")
    kw3_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['kw3'][1]], name="kw3_val")

    topic1_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['topic1'][1]], name="topic1_index")
    topic1_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['topic1'][1]], name="topic1_val")
    topic2_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['topic2'][1]], name="topic2_index")
    topic2_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['topic2'][1]], name="topic2_val")
    topic3_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['topic3'][1]], name="topic3_index")
    topic3_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['topic3'][1]], name="topic3_val")

    aid_p = tf.placeholder(tf.int32, [None, 1], name="aid")
    advertiserid_p = tf.placeholder(tf.int32, [None, 1], name="advertiserId")
    campaignid_p = tf.placeholder(tf.int32, [None, 1], name="campaignId")
    creativeid_p = tf.placeholder(tf.int32, [None, 1], name="creativeId")
    adcategoryid_p = tf.placeholder(tf.int32, [None, 1], name="adCategoryId")
    productid_p = tf.placeholder(tf.int32, [None, 1], name="productId")
    producttype_p = tf.placeholder(tf.int32, [None, 1], name="productType")

    true_label = tf.placeholder(tf.float32, [None, 1], name="true_label")
    # ****************************************************************** place holder end

    pred_val, model_loss, network_params = inference(uid_p, lbs_p, age_p, carrier_p, consumptionability_p, education_p,
                                                     gender_p, house_p, os_p, ct_p, marriagestatus_index_p, marriagestatus_val_p, appidaction_index_p, appidaction_val_p, appIdInstall_index_p,
                                                     appIdInstall_val_p, interest1_index_p, interest1_val_p, interest2_index_p, interest2_val_p, interest3_index_p, interest3_val_p, interest4_index_p,
                                                     interest4_val_p, interest5_index_p, interest5_val_p, kw1_index_p, kw1_val_p, kw2_index_p, kw2_val_p,
                                                     kw3_index_p, kw3_val_p, topic1_index_p, topic1_val_p, topic2_index_p, topic2_val_p, topic3_index_p,
                                                     topic3_val_p, aid_p, advertiserid_p, campaignid_p, creativeid_p, adcategoryid_p, productid_p, producttype_p, creativesize_p, true_label, feature_conf_dict,
                                                     graph_hyper_params)


    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_step = None
    learning_rate = tf.Variable(float(graph_hyper_params['learn_rate']), trainable=False, dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.5)
    if graph_hyper_params['opt'] == 'adam':
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(model_loss, global_step=global_step)
    elif graph_hyper_params['opt'] == 'adgrad':
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(model_loss, global_step=global_step)
    elif graph_hyper_params['opt'] == 'adadelta':
        train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(model_loss, global_step=global_step)
    elif graph_hyper_params['opt'] == 'ftrl':
        train_step = tf.train.FtrlOptimizer(learning_rate).minimize(model_loss, global_step=global_step)
    elif graph_hyper_params['opt'] == 'sgd':
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(model_loss, global_step=global_step)
    else:
        print 'No optimizer !'

    time_now = 'model_' + str(graph_hyper_params['model']) + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    checkpoint_dir = os.path.abspath("./checkpoints/dmf_tencent/" + time_now)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


    def get_fed_dict(b_data, split_vector_data, feature_conf_dict, user_data_in, user_data_in_map_row, predict=False):
        if graph_hyper_params['formal']:
            aid_list = b_data['aid'].values
            uid_list = b_data['uid'].values
        else:
            if len(b_data) == 4:
                aid_list, uid_list = [1, 2, 3, 4], [1, 2, 3, 4]
            elif len(b_data) == 3:
                aid_list, uid_list = [1, 2, 3], [1, 2, 3]
            elif len(b_data) == 2:
                aid_list, uid_list = [1, 2], [1, 2]
            else:
                aid_list, uid_list = [1], [1]

        # print 11
        # d1 = datetime.now()
        b_u_d, b_a_d = [], []
        for b_uid in uid_list:
            b_u_d.append(user_data_in.iloc[user_data_in_map_row[b_uid]])
            # if predict == 0:
            #     b_u_d.append(user_data_train.iloc[uid_map_row_train[b_uid]])
            # elif predict == 1:
            #     b_u_d.append(user_data_predict1.iloc[uid_map_row_predict_1[b_uid]])
            # elif predict == 2:
            #     b_u_d.append(user_data_predict2.iloc[uid_map_row_predict_2[b_uid]])
            # else:
            #     print 'fed wrong!'
        for b_aid in aid_list:
            b_a_d.append(ad_data.iloc[aid_map_row[b_aid]])
        b_u_d = pd.concat(b_u_d, axis=1).transpose()
        b_a_d = pd.concat(b_a_d, axis=1).transpose()
        # d3 = datetime.now()

        # print 12
        # pd.concat([data.iloc[1].to_frame(), data.iloc[2].to_frame()], axis=1).transpose()
        fed_dict = {}
        fed_dict[uid_p] = np.expand_dims(b_u_d['uid'], axis=1)
        fed_dict[lbs_p] = np.expand_dims(b_u_d['LBS'], axis=1)
        fed_dict[age_p] = np.expand_dims(b_u_d['age'], axis=1)
        fed_dict[carrier_p] = np.expand_dims(b_u_d['carrier'], axis=1)
        fed_dict[consumptionability_p] = np.expand_dims(b_u_d['consumptionAbility'], axis=1)
        fed_dict[education_p] = np.expand_dims(b_u_d['education'], axis=1)
        fed_dict[gender_p] = np.expand_dims(b_u_d['gender'], axis=1)
        fed_dict[house_p] = np.expand_dims(b_u_d['house'], axis=1)
        fed_dict[os_p] = np.expand_dims(b_u_d['os'], axis=1)
        fed_dict[ct_p] = np.expand_dims(b_u_d['ct'], axis=1)
        # fed_dict[marriagestatus_p] = np.expand_dims(b_u_d['marriageStatus'], axis=1)
        # print 121
        appidaction_li = split_vector_data(b_u_d['appIdAction'])
        # print 1212
        fed_dict[appidaction_index_p], fed_dict[appidaction_val_p] = appidaction_li[0], appidaction_li[1]
        appIdInstall_li = split_vector_data(b_u_d['appIdInstall'])
        fed_dict[appIdInstall_index_p], fed_dict[appIdInstall_val_p] = appIdInstall_li[0], appIdInstall_li[1]
        # print 122
        marriagestatus_li = split_vector_data(b_u_d['marriageStatus'], interest='marriageStatus', feature_config=feature_conf_dict)
        fed_dict[marriagestatus_index_p], fed_dict[marriagestatus_val_p] = marriagestatus_li[0], marriagestatus_li[1]
        interest1_li = split_vector_data(b_u_d['interest1'], interest='interest1', feature_config=feature_conf_dict)
        fed_dict[interest1_index_p], fed_dict[interest1_val_p]  = interest1_li[0], interest1_li[1]
        interest2_li = split_vector_data(b_u_d['interest2'], interest='interest2', feature_config=feature_conf_dict)
        fed_dict[interest2_index_p], fed_dict[interest2_val_p] = interest2_li[0], interest2_li[1]
        interest3_li = split_vector_data(b_u_d['interest3'], interest='interest3', feature_config=feature_conf_dict)
        fed_dict[interest3_index_p], fed_dict[interest3_val_p] = interest3_li[0], interest3_li[1]
        interest4_li = split_vector_data(b_u_d['interest4'], interest='interest4', feature_config=feature_conf_dict)
        fed_dict[interest4_index_p], fed_dict[interest4_val_p] = interest4_li[0], interest4_li[1]
        interest5_li = split_vector_data(b_u_d['interest5'], interest='interest5', feature_config=feature_conf_dict)
        fed_dict[interest5_index_p], fed_dict[interest5_val_p] = interest5_li[0], interest5_li[1]
        # print 123
        kw1_li = split_vector_data(b_u_d['kw1'])
        fed_dict[kw1_index_p], fed_dict[kw1_val_p] = kw1_li[0], kw1_li[1]
        kw2_li = split_vector_data(b_u_d['kw2'])
        fed_dict[kw2_index_p], fed_dict[kw2_val_p] = kw2_li[0], kw2_li[1]
        kw3_li = split_vector_data(b_u_d['kw3'])
        fed_dict[kw3_index_p], fed_dict[kw3_val_p] = kw3_li[0], kw3_li[1]
        # print 124
        topic1_li = split_vector_data(b_u_d['topic1'])
        fed_dict[topic1_index_p], fed_dict[topic1_val_p] = topic1_li[0], topic1_li[1]
        topic2_li = split_vector_data(b_u_d['topic2'])
        fed_dict[topic2_index_p], fed_dict[topic2_val_p] = topic2_li[0], topic2_li[1]
        topic3_li = split_vector_data(b_u_d['topic3'])
        fed_dict[topic3_index_p], fed_dict[topic3_val_p] = topic3_li[0], topic3_li[1]
        # print 125

        # # ad
        fed_dict[aid_p] = np.expand_dims(b_a_d['aid'], axis=1)
        fed_dict[advertiserid_p] = np.expand_dims(b_a_d['advertiserId'], axis=1)
        fed_dict[campaignid_p] = np.expand_dims(b_a_d['campaignId'], axis=1)
        fed_dict[creativeid_p] = np.expand_dims(b_a_d['creativeId'], axis=1)
        fed_dict[adcategoryid_p] = np.expand_dims(b_a_d['adCategoryId'], axis=1)
        fed_dict[productid_p] = np.expand_dims(b_a_d['productId'], axis=1)
        fed_dict[producttype_p] = np.expand_dims(b_a_d['productType'], axis=1)

        # print 13
        # fed_dict[creativesize_p] = np.expand_dims(b_a_d['creativeSize'], axis=1)
        if graph_hyper_params['creativeSize_pro'] == 'min_max':
            fed_dict[creativesize_p] = np.expand_dims(b_a_d['creativeSize'], axis=1).astype(np.float32)
        elif graph_hyper_params['creativeSize_pro'] == 'li_san':
            fed_dict[creativesize_p] = np.expand_dims(b_a_d['creativeSize'], axis=1)
        else:
            print 'wrong feed'

        # label
        # print 14
        if not predict:
            fed_dict[true_label] = np.expand_dims(b_data['label'].values, axis=1).astype(np.float32)
        # print 15
        # d4 = datetime.now()
        # print d2-d1, d3-d2, d4-d3
        # print fed_dict[true_label]
        # print len(fed_dict[true_label]), len(fed_dict[aid_p]), len(fed_dict[uid_p]),
        return fed_dict

    # def eval_on_dev(split_vector_data):
    #     e_b_s = len(dev_data) / graph_hyper_params['batch_size']
    #     auc_true, auc_pre = [], []
    #     # auc = []
    #     for index in tqdm(range(e_b_s)):
    #         start = index * graph_hyper_params['batch_size']
    #         end = (index + 1) * graph_hyper_params['batch_size'] if (index + 1) * graph_hyper_params['batch_size'] < len(dev_data) else len(dev_data)
    #         b_dev_data = dev_data[start:end]
    #         fed_dict = get_fed_dict(b_dev_data, split_vector_data, feature_conf_dict)
    #         pred_value, pre_pred_value, final_vec, uu, vv = sess.run([pred_val, network_params[0], network_params[1], network_params[2], network_params[3]], feed_dict=fed_dict)
    #
    #         pre_real_val = np.array(pred_value).reshape((-1))
    #         auc_true = auc_true + list(b_dev_data['label'].values)
    #         auc_pre = auc_pre + pre_real_val.tolist()
    #
    #         if True in np.isnan(pre_real_val):
    #             print 'contain nan: ', np.array(pre_pred_value).reshape((-1))
    #             print np.array(final_vec).reshape((-1))
    #             print np.array(uu).reshape((-1))
    #             print np.array(vv).reshape((-1))
    #
    #         # auc.append()
    #     # auc_pre = np.array(auc_pre)
    #     # auc_pre = np.exp(auc_pre) / np.exp(auc_pre).sum()
    #     # print auc_true
    #     # print auc_pre
    #     fpr, tpr, thresholds = metrics.roc_curve(auc_true, auc_pre, pos_label=1)
    #     auc_v, gni = metrics.auc(fpr, tpr), gini_norm(auc_true, auc_pre)
    #
    #     auc_pre_2 = np.array(auc_pre)
    #     auc_pre_2.sort()
    #     print('dev_pre_top2=%.4f %.4f min2=%.4f %.4f' %
    #           (auc_pre_2.tolist()[-1], auc_pre_2.tolist()[-2], auc_pre_2.tolist()[0], auc_pre_2.tolist()[1]))
    #     return auc_v, gni


    best_auc = 0.0
    split_vector_data = SplitClass()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(graph_hyper_params['epoch']): # 只训练 1 轮
        e_b_s = len(train_data) / graph_hyper_params['batch_size']
        one_epoch_loss, one_epoch_batchnum = 0.0, 0.0
        for index in tqdm(range(e_b_s)):
            # print 0
            start = index * graph_hyper_params['batch_size']
            end = (index + 1) * graph_hyper_params['batch_size'] if (index + 1) * graph_hyper_params['batch_size'] < len(train_data) else len(train_data)
            b_data = train_data[start:end]

            # print 1
            # d1 = datetime.now()
            fed_dict = get_fed_dict(b_data, split_vector_data, feature_conf_dict, user_data_train, uid_map_row_train)
            # d2 = datetime.now()
            # print 2
            _, loss_val, pre_tr_val = sess.run([train_step, model_loss, network_params[0]], feed_dict=fed_dict)
            # print 3
            # d3 = datetime.now()
            # print d2-d1, d3-d2
            one_epoch_loss += loss_val
            one_epoch_batchnum += 1.

            if graph_hyper_params['debug']:
                print datetime.now(), index, loss_val
            pre_tr_val = np.array(pre_tr_val).reshape((-1))
            if graph_hyper_params['debug'] or True in np.isnan(pre_tr_val):
                print pre_tr_val

            if index != 0 and index % ((e_b_s - 1) / graph_hyper_params['show_peroid']) == 0:
                split_vector_data.clean()
            #     auc, gn = eval_on_dev(split_vector_data)
            #     best_auc = max(auc, best_auc)
            #     format_str = '%s epoch=%.2f avg_loss=%.4f auc=%.4f best_auc=%.4f gn=%.4f'
            #     print (format_str % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (epoch + 1.0 * (index+1) / e_b_s), one_epoch_loss / one_epoch_batchnum, auc, best_auc, gn))
            #     one_epoch_loss = one_epoch_batchnum = 0.0


    # pass
    del split_vector_data, user_data_train, train_data
    gc.collect()
    split_vector_data = None
    if graph_hyper_params['test1']:
        predict_data = predict_data1.sort_values(by='uid')
        if graph_hyper_params['formal']:
            graph_hyper_params['batch_size'] = 512
        e_b_s = len(predict_data) / graph_hyper_params['batch_size'] if len(predict_data) % graph_hyper_params[
            'batch_size'] == 0 else len(predict_data) / graph_hyper_params['batch_size'] + 1
        split_vector_data = SplitClass()
        pred = []
        for index in tqdm(range(e_b_s)):
            start = index * graph_hyper_params['batch_size']
            end = (index + 1) * graph_hyper_params['batch_size'] if (index + 1) * graph_hyper_params['batch_size'] <= len(predict_data) else len(predict_data) + 1
            b_predict_data = predict_data[start:end]
            # print len(b_predict_data), start, end
            # fed_dict = get_fed_dict(b_dev_data, split_vector_data, feature_conf_dict)
            fed_dict = get_fed_dict(b_predict_data, split_vector_data, feature_conf_dict, user_data_predict1, uid_map_row_predict_1, predict=True)
            pred_value = sess.run([pred_val], feed_dict=fed_dict)
            # print pred_value
            pre_real_val = np.array(pred_value).reshape((-1))
            pred = pred + pre_real_val.tolist()

            if graph_hyper_params['formal'] and index != 0 and index % ((e_b_s - 1) / graph_hyper_params['show_peroid']) == 0:
                split_vector_data.clean()

        print len(predict_data), len(pred)
        predict_data['pred_label'] = pred
        csv_data = predict_data[['ori_aid', 'ori_uid', 'pred_label']]
        csv_data.columns = ['aid', 'uid', 'score']
        csv_path = os.path.join(checkpoint_dir, 'test1_'+'n' + str(graph_hyper_params['neg_start']) + '_submission.csv')
        csv_data.to_csv(csv_path, index=False)
        print 'submission_path:', csv_path


    del split_vector_data, user_data_predict1, predict_data1
    gc.collect()
    if graph_hyper_params['test2']:
        predict_data = predict_data2.sort_values(by='uid')
        if graph_hyper_params['formal']:
            graph_hyper_params['batch_size'] = 512
        e_b_s = len(predict_data) / graph_hyper_params['batch_size'] if len(predict_data) % graph_hyper_params[
            'batch_size'] == 0 else len(predict_data) / graph_hyper_params['batch_size'] + 1
        split_vector_data = SplitClass()
        # split_vector_data.clean()
        pred = []
        for index in tqdm(range(e_b_s)):
            start = index * graph_hyper_params['batch_size']
            end = (index + 1) * graph_hyper_params['batch_size'] if (index + 1) * graph_hyper_params[
                'batch_size'] <= len(predict_data) else len(predict_data) + 1
            b_predict_data = predict_data[start:end]
            # print len(b_predict_data), start, end
            # fed_dict = get_fed_dict(b_dev_data, split_vector_data, feature_conf_dict)
            fed_dict = get_fed_dict(b_predict_data, split_vector_data, feature_conf_dict, user_data_predict2, uid_map_row_predict_2, predict=True)
            pred_value = sess.run([pred_val], feed_dict=fed_dict)
            # print pred_value
            pre_real_val = np.array(pred_value).reshape((-1))
            pred = pred + pre_real_val.tolist()

            if graph_hyper_params['formal'] and index != 0 and index % ((e_b_s - 1) / graph_hyper_params['show_peroid']) == 0:
                split_vector_data.clean()

        print len(predict_data), len(pred)
        predict_data['pred_label'] = pred
        csv_data = predict_data[['ori_aid', 'ori_uid', 'pred_label']]
        csv_data.columns = ['aid', 'uid', 'score']
        csv_path = os.path.join(checkpoint_dir, 'test2_' + 'n' + str(graph_hyper_params['neg_start']) + '_submission.csv')
        csv_data.to_csv(csv_path, index=False)
        print 'submission_path:', csv_path
    pass


def parse():
    args = argparse.ArgumentParser(description='Ten Con !')
    args.add_argument('--model', type=str, default='dmf', help='model type')
    args.add_argument('--opt', type=str, default='adam', help='opt')
    args.add_argument('--lr', type=float, default=0.0001, help='lr')
    args.add_argument('--ns', type=int, default=1, help='neg size')
    args.add_argument('--only_train', type=int, default=1, help='only train')
    args.add_argument('--shp', type=int, default=10, help='show peroid')
    args.add_argument('--neg_start', type=int, default=0, help='neg_start') # neg_start

    args.add_argument('--test1', type=int, default=0, help='test1') # neg_start
    args.add_argument('--test2', type=int, default=1, help='test2') # neg_start
    return args


if __name__ == '__main__':
    args = parse()
    args = args.parse_args()
    # print args.lr
    # min_max li_san
    # graph_hyper_params = {'batch_size': 4, 'l2_reg_alpha':0.0, 'learn_rate': args.lr, 'show_peroid': 2,
    #                       'formal': False, 'epoch': 2, 'debug': True, 'o_dev_size': 2,
    #                       'creativeSize_pro': 'li_san', 'neg_size': 1, 'model': args.model, 'opt': args.opt,
    #                       'neg_start': args.neg_start, 'only_train': True, 'test1':args.test1, 'test2':args.test2}


    graph_hyper_params = {'batch_size': 128, 'l2_reg_alpha':0.0, 'learn_rate': args.lr, 'show_peroid': args.shp,
                          'formal': True, 'epoch': 1, 'debug': False, 'o_dev_size': 10000,
                          'creativeSize_pro': 'li_san', 'neg_size': args.ns, 'model': args.model, 'opt': args.opt,
                          'neg_start': args.neg_start, 'only_train': False, 'test1':args.test1, 'test2':args.test2}


    train_eval_model(graph_hyper_params)
    pass
