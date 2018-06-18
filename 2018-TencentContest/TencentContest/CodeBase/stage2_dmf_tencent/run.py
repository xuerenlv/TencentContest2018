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

from metrics import gini_norm
from models import inference
from read_data import get_prod_dataset
from sklearn import cross_validation,metrics
from tqdm import tqdm

# one_hot_feature = ['aid', 'uid', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education',
#                    'gender', 'house', 'os', 'ct', 'marriageStatus', 'advertiserId',
#                    'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']
# vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
#                   'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
# continuous_feature = 'creativeSize'

user_features = ['uid', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education',
                 'gender', 'house', 'os', 'ct', 'marriageStatus',
                 'appIdAction', 'appIdInstall', 'interest1', 'interest2',
                 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
ad_features = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType',
               'creativeSize']

class SplitClass:
    def __init__(self):
        self.split_cache, self.split_cache_rem_size = {}, {}

    def clean(self):
        d_key = []
        for k in self.split_cache_rem_size:
            if self.split_cache_rem_size[k] <= 10:
                d_key.append(k)
        for k in d_key:
            del self.split_cache[k], self.split_cache_rem_size[k]
        pass

    def __call__(self, vda):
        index_data, val_data = [], [] #, , len_data = [], [], []
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
        if x == -100:
            return 0
        if x not in self.d:
            self.d[x] = len(self.d) + 1
        return self.d[x]

def train_eval_model(graph_hyper_params):
    # global pos_train_data, neg_train_data, dev_data, predict_data, relevant_user_data, no_relevant_user_data, ad_data, feature_conf_dict
    all_train_data, dev_data, predict_data, relevant_user_data, no_relevant_user_data, ad_data, feature_conf_dict, re_uid_map, re_aid_map = get_prod_dataset(graph_hyper_params['formal'])
    print graph_hyper_params


    # 重新 split train dev
    o_dev_size = graph_hyper_params['o_dev_size']
    atd = pd.concat([all_train_data, dev_data])
    pos_atd, neg_atd = atd[atd['label'] == 1], atd[atd['label'] == 0]
    dev_data = pd.concat([pos_atd[:o_dev_size], neg_atd[:o_dev_size]])
    pos_train_data, neg_train_data = pos_atd[o_dev_size:], neg_atd[o_dev_size:]
    print 'dev_size', len(dev_data)
    print 'pos-neg-all', len(pos_train_data), len(neg_train_data), len(all_train_data)
    del all_train_data
    gc.collect()
    # **********************************

    print 'map row start'
    uid_map_row, aid_map_row = dict(zip(relevant_user_data['uid'].values, np.arange(len(relevant_user_data)))), dict(zip(ad_data['aid'].values, np.arange(len(ad_data))))
    print 'map row end'

    # 对 creativeSize 这一个连续特征的处理
    if graph_hyper_params['creativeSize_pro'] == 'min_max':
        print 'min-max norm creativeSize', ad_data['creativeSize'].max(), ad_data['creativeSize'].min()
        norm_cs = (ad_data['creativeSize'] * 1.0 - ad_data['creativeSize'].min()) / (ad_data['creativeSize'].max() - ad_data['creativeSize'].min())
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

    print feature_conf_dict
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
    marriagestatus_p = tf.placeholder(tf.int32, [None, 1], name="marriageStatus")

    appidaction_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['appIdAction'][1]], name="appidaction_index")
    appidaction_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['appIdAction'][1]], name="appidaction_val")
    appIdInstall_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['appIdInstall'][1]], name="appIdInstall_index")
    appIdInstall_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['appIdInstall'][1]], name="appIdInstall_val")

    interest1_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['interest1'][1]], name="interest1_index")
    interest1_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['interest1'][1]], name="interest1_val")
    interest2_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['interest2'][1]], name="interest2_index")
    interest2_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['interest2'][1]], name="interest2_val")
    interest3_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['interest3'][1]], name="interest3_index")
    interest3_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['interest3'][1]], name="interest3_val")
    interest4_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['interest4'][1]], name="interest4_index")
    interest4_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['interest4'][1]], name="interest4_val")
    interest5_index_p = tf.placeholder(tf.int32, [None, feature_conf_dict['interest5'][1]], name="interest5_index")
    interest5_val_p = tf.placeholder(tf.float32, [None, 1, feature_conf_dict['interest5'][1]], name="interest5_val")

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

    train_p = tf.placeholder(tf.bool, name="train_p")
    dropout_p = tf.placeholder(tf.float32, shape=[None], name="dropout_p")
    # ****************************************************************** place holder end

    pred_val, model_loss, network_params = inference(uid_p, lbs_p, age_p, carrier_p, consumptionability_p, education_p,
                                                     gender_p, house_p, os_p, ct_p, marriagestatus_p, appidaction_index_p, appidaction_val_p, appIdInstall_index_p,
                                                     appIdInstall_val_p, interest1_index_p, interest1_val_p, interest2_index_p, interest2_val_p, interest3_index_p, interest3_val_p, interest4_index_p,
                                                     interest4_val_p, interest5_index_p, interest5_val_p, kw1_index_p, kw1_val_p, kw2_index_p, kw2_val_p,
                                                     kw3_index_p, kw3_val_p, topic1_index_p, topic1_val_p, topic2_index_p, topic2_val_p, topic3_index_p,
                                                     topic3_val_p, aid_p, advertiserid_p, campaignid_p, creativeid_p, adcategoryid_p, productid_p, producttype_p, creativesize_p, true_label, feature_conf_dict,
                                                     graph_hyper_params, train_p, dropout_p)

    # pred_val_for_pre, _, __ = inference(uid_p, lbs_p, age_p, carrier_p, consumptionability_p, education_p,
    #                                                  gender_p, house_p, os_p, ct_p, marriagestatus_p, appidaction_index_p, appidaction_val_p, appIdInstall_index_p,
    #                                                  appIdInstall_val_p, interest1_index_p, interest1_val_p, interest2_index_p, interest2_val_p, interest3_index_p, interest3_val_p, interest4_index_p,
    #                                                  interest4_val_p, interest5_index_p, interest5_val_p, kw1_index_p, kw1_val_p, kw2_index_p, kw2_val_p,
    #                                                  kw3_index_p, kw3_val_p, topic1_index_p, topic1_val_p, topic2_index_p, topic2_val_p, topic3_index_p,
    #                                                  topic3_val_p, aid_p, advertiserid_p, campaignid_p, creativeid_p, adcategoryid_p, productid_p, producttype_p, creativesize_p, true_label, feature_conf_dict, graph_hyper_params, istrain=False)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_step = None
    if graph_hyper_params['opt'] == 'adam':
        train_step = tf.train.AdamOptimizer(graph_hyper_params['learn_rate']).minimize(model_loss, global_step=global_step)
    elif graph_hyper_params['opt'] == 'adgrad':
        train_step = tf.train.AdagradOptimizer(graph_hyper_params['learn_rate']).minimize(model_loss, global_step=global_step)
    elif graph_hyper_params['opt'] == 'adadelta':
        train_step = tf.train.AdadeltaOptimizer(graph_hyper_params['learn_rate']).minimize(model_loss, global_step=global_step)
    else:
        print 'No optimizer !'

    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_dir = os.path.abspath("./checkpoints/dmf_tencent/" + time_now)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def get_fed_dict(b_data, split_vector_data):
        if graph_hyper_params['formal']:
            aid_list = b_data['aid'].values
            uid_list = b_data['uid'].values
        else:
            if len(b_data) == 4:
                aid_list, uid_list = [11, 11, 11, 11], [11, 190, 191, 11]
            elif len(b_data) == 3:
                aid_list, uid_list = [11, 11, 11], [11, 190, 191]
            else:
                aid_list, uid_list = [11], [11]

        # print 11
        # d1 = datetime.now()
        b_u_d, b_a_d = [], []
        for b_uid in uid_list:
            b_u_d.append(relevant_user_data.iloc[uid_map_row[b_uid]])
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
        fed_dict[marriagestatus_p] = np.expand_dims(b_u_d['marriageStatus'], axis=1)
        # print 121
        appidaction_li = split_vector_data(b_u_d['appIdAction'])
        # print 1212
        fed_dict[appidaction_index_p], fed_dict[appidaction_val_p] = appidaction_li[0], appidaction_li[1]
        appIdInstall_li = split_vector_data(b_u_d['appIdInstall'])
        fed_dict[appIdInstall_index_p], fed_dict[appIdInstall_val_p] = appIdInstall_li[0], appIdInstall_li[1]
        # print 122
        interest1_li = split_vector_data(b_u_d['interest1'])
        fed_dict[interest1_index_p], fed_dict[interest1_val_p]  = interest1_li[0], interest1_li[1]
        interest2_li = split_vector_data(b_u_d['interest2'])
        fed_dict[interest2_index_p], fed_dict[interest2_val_p] = interest2_li[0], interest2_li[1]
        interest3_li = split_vector_data(b_u_d['interest3'])
        fed_dict[interest3_index_p], fed_dict[interest3_val_p] = interest3_li[0], interest3_li[1]
        interest4_li = split_vector_data(b_u_d['interest4'])
        fed_dict[interest4_index_p], fed_dict[interest4_val_p] = interest4_li[0], interest4_li[1]
        interest5_li = split_vector_data(b_u_d['interest5'])
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
        fed_dict[true_label] = np.expand_dims(b_data['label'].values, axis=1).astype(np.float32)
        # print 15
        # d4 = datetime.now()
        # print d2-d1, d3-d2, d4-d3
        # print fed_dict[true_label]
        # print len(fed_dict[true_label]), len(fed_dict[aid_p]), len(fed_dict[uid_p]),
        return fed_dict

    def eval_on_dev(split_vector_data):
        e_b_s = len(dev_data) / graph_hyper_params['batch_size']
        auc_true, auc_pre = [], []
        # auc = []
        for index in tqdm(range(e_b_s)):
            start = index * graph_hyper_params['batch_size']
            end = (index + 1) * graph_hyper_params['batch_size'] if (index + 1) * graph_hyper_params['batch_size'] < len(dev_data) else len(dev_data)
            b_dev_data = dev_data[start:end]
            fed_dict = get_fed_dict(b_dev_data, split_vector_data)
            fed_dict[train_p] = False
            fed_dict[dropout_p] = np.array([1.0])
            pred_value = sess.run([pred_val], feed_dict=fed_dict)

            pre_real_val = np.array(pred_value).reshape((-1))
            auc_true = auc_true + list(b_dev_data['label'].values)
            auc_pre = auc_pre + pre_real_val.tolist()
            # auc.append()
        # auc_pre = np.array(auc_pre)
        # auc_pre = np.exp(auc_pre) / np.exp(auc_pre).sum()
        # print auc_true
        # print auc_pre
        fpr, tpr, thresholds = metrics.roc_curve(auc_true, auc_pre, pos_label=1)
        # >> > metrics.auc(fpr, tpr)
        return metrics.auc(fpr, tpr), gini_norm(auc_true, auc_pre)

    # def predict_csv(split_vector_data):
    #     e_b_s = len(predict_data) / graph_hyper_params['batch_size'] if len(predict_data) % graph_hyper_params['batch_size']==0 else len(predict_data) / graph_hyper_params['batch_size'] + 1
    #     pred = []
    #     for index in tqdm(range(e_b_s)):
    #         start = index * graph_hyper_params['batch_size']
    #         end = (index + 1) * graph_hyper_params['batch_size'] if (index + 1) * graph_hyper_params['batch_size'] < len(predict_data) else len(predict_data)+1
    #         b_predict_data = predict_data[start:end]
    #         # print len(b_predict_data), start, end
    #         fed_dict = get_fed_dict(b_predict_data, split_vector_data)
    #         pred_value = sess.run([pred_val], feed_dict=fed_dict)
    #         pre_real_val = np.array(pred_value).reshape((-1))
    #         pred = pred + pre_real_val.tolist()
    #     # print len(pred), len(predict_data)
    #     predict_data['pred_label'] = pred
    #     csv_data = predict_data[['ori_aid', 'ori_uid', 'pred_label']]
    #     csv_data.columns = ['aid', 'uid', 'score']
    #     csv_path = checkpoint_dir+'/submission.csv'
    #     csv_data.to_csv(csv_path, index=False)
    #     return csv_path


    def save_predict_material(user_data, ad_data):
        user_data_file = os.path.join(checkpoint_dir, 'user_data_file.csv')
        ad_data_file = os.path.join(checkpoint_dir, 'ad_data_file.csv')
        graph_hyper_params_file = os.path.join(checkpoint_dir, 'graph_hyper_params_file.pic')

        user_data.to_csv(user_data_file, index=False)
        ad_data.to_csv(ad_data_file, index=False)
        pickle.dump(graph_hyper_params, open(graph_hyper_params_file, 'w'))
        pass

    def construct_train_data(start_neg, pos_train_data, neg_train_data, graph_hyper_params):
        # global pos_train_data, neg_train_data, start_neg
        pos_len, neg_len = len(pos_train_data), len(neg_train_data)
        # print start_neg, pos_len, neg_len
        if start_neg + pos_len < neg_len:
            this_neg_train_data = neg_train_data[start_neg : start_neg + graph_hyper_params['neg_size']*pos_len]
            start_neg += pos_len*graph_hyper_params['neg_size']
        else:
            this_neg_train_data = pd.concat([neg_train_data[start_neg : neg_len], neg_train_data[0 : graph_hyper_params['neg_size']*pos_len - (neg_len-start_neg)]])
            start_neg = graph_hyper_params['neg_size']*pos_len - (neg_len-start_neg)
        train_data = pd.concat([pos_train_data, this_neg_train_data])
        return shuffle(train_data), start_neg

    best_auc = 0.0
    start_neg = 0
    split_vector_data = SplitClass()
    save_data_for_predict = False
    for epoch in range(graph_hyper_params['epoch']):
        train_data, start_neg = construct_train_data(start_neg, pos_train_data, neg_train_data, graph_hyper_params)
        if start_neg < graph_hyper_params['neg_size'] * len(pos_train_data):
            neg_train_data = shuffle(neg_train_data)

        e_b_s = len(train_data) / graph_hyper_params['batch_size']
        one_epoch_loss, one_epoch_batchnum = 0.0, 0.0
        early_stop_hit = 0
        split_vector_data.clean()
        for index in tqdm(range(e_b_s)):
            # print 0
            start = index * graph_hyper_params['batch_size']
            end = (index + 1) * graph_hyper_params['batch_size'] if (index + 1) * graph_hyper_params['batch_size'] < len(train_data) else len(train_data)
            b_data = train_data[start:end]

            # print 1
            # d1 = datetime.now()
            fed_dict = get_fed_dict(b_data, split_vector_data)
            fed_dict[train_p] = True
            fed_dict[dropout_p] = np.array([graph_hyper_params['dropout_keep']])
            # d2 = datetime.now()
            # print 2
            _, loss_val = sess.run([train_step, model_loss], feed_dict=fed_dict)
            # print 3
            # d3 = datetime.now()
            # print d2-d1, d3-d2
            one_epoch_loss += loss_val
            one_epoch_batchnum += 1.0

            if graph_hyper_params['debug']:
                print datetime.now(), index, loss_val

            if index != 0 and index % ((e_b_s - 1) / graph_hyper_params['show_peroid']) == 0:
                auc, gn = eval_on_dev(split_vector_data)
                best_auc = max(auc, best_auc)
                format_str = '%s epoch=%.2f avg_loss=%.4f auc=%.4f best_auc=%.4f gn=%.4f'
                print (format_str % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (epoch + 1.0 * (index+1) / e_b_s), one_epoch_loss / one_epoch_batchnum, auc, best_auc, gn))
                one_epoch_loss = one_epoch_batchnum = 0.0

                # global split_cache, split_cache_rem_size
                # if len(split_cache) > 10000000:
                #     keys = split_cache.keys()
                #     for key in keys:
                #         if split_cache_rem_size[key] < 2:
                #             del split_cache_rem_size[key], split_cache[key]


                if (auc >= best_auc and (epoch + 1.0 * (index+1) / e_b_s) >= 0.6 and auc > 0.72) or (auc >= best_auc and auc>0.75):
                    current_step = tf.train.global_step(sess, global_step)
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("saved model to: %s" % path)

                    if not save_data_for_predict:
                        udp = pd.concat([relevant_user_data, no_relevant_user_data])
                        save_predict_material(udp, ad_data)
                        save_data_for_predict = True
                    early_stop_hit = 0
                elif auc < best_auc and abs(auc-best_auc) > 0.02:
                    early_stop_hit += 1
                    if early_stop_hit >= 3:
                        print 'eary_stop_best:', best_auc
                        import sys
                        sys.exit(0)

                    # csv_path = predict_csv(split_vector_data)
                    # print 'save csv to: ', csv_path
    pass


def parse():
    args = argparse.ArgumentParser(description='Ten Con !')
    args.add_argument('--model', type=str, default='dmf', help='model type')
    args.add_argument('--opt', type=str, default='adam', help='opt')
    args.add_argument('--lr', type=float, default=0.0001, help='lr')
    args.add_argument('--ns', type=int, default=1, help='neg size')
    args.add_argument('--l2', type=float, default=0.0, help='l2')
    args.add_argument('--uk', type=bool, default=False, help='l2')
    return args


if __name__ == '__main__':
    args = parse()
    args = args.parse_args()
    # print args.lr
    # min_max li_san
    # graph_hyper_params = {'batch_size': 4, 'l2_reg_alpha': args.l2, 'learn_rate': args.lr, 'show_peroid': 1,
    #                       'formal': False, 'epoch': 10, 'debug': True, 'o_dev_size': 5,
    #                       'creativeSize_pro': 'li_san', 'neg_size': 1, 'model': args.model, 'opt': args.opt,
    #                       'use_kernal': True, 'dmf_det': True, 'dropout_keep':0.5, 'use_bn': True}

    graph_hyper_params = {'batch_size': 128, 'l2_reg_alpha': args.l2, 'learn_rate': args.lr, 'show_peroid': 10,
                          'formal': True, 'epoch': 50, 'debug': False, 'o_dev_size': 10000,
                          'creativeSize_pro': 'li_san', 'neg_size': args.ns, 'model': args.model, 'opt':args.opt,
                          'use_kernal': args.uk, 'dmf_det': True, 'dropout_keep': 0.5, 'use_bn': True}


    train_eval_model(graph_hyper_params)
    pass
