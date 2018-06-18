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

from run import SplitClass


# def get_predict_data(formal):
#     if formal:
#         dir_name = '../../DataSet/preliminary_contest_data/'
#     else:
#         dir_name = '../../DataSet/small_preliminary_contest_data/'
#     # train_data_file = dir_name + 'finally_processed_data_train.csv'
#     # dev_data_file = dir_name + 'finally_processed_data_dev.csv'
#     predict_data_file = dir_name + 'finally_processed_data_predict.csv'
#     relevant_user_data_file = dir_name + 'finally_processed_data_user_relevant.csv'
#     # no_relevant_user_data_file = dir_name + 'finally_processed_data_user_no_rel.csv'
#     ad_data_file = dir_name + 'finally_processed_data_ad.csv'
#     # feature_conf_dict_file = dir_name + 'finally_feature_conf_dict.pic'
#     predict_data, relevant_user_data = pd.read_csv(predict_data_file), pd.read_csv(relevant_user_data_file)
#     user_data = relevant_user_data[relevant_user_data['uid'].isin(predict_data['uid'].unique())]
#     return predict_data, user_data

def get_map_dict_and_predict(formal): # newid -> oriid
    if formal:
        dir_name = '../../DataSet/preliminary_contest_data/'
    else:
        dir_name = '../../DataSet/small_preliminary_contest_data/'
    test_file = dir_name + 'test1.csv'
    re_uid_map_file = dir_name + 'finally_re_uid_map_dict.pic'
    re_aid_map_file = dir_name + 'finally_re_aid_map_dict.pic'
    re_uid_map = pickle.load(open(re_uid_map_file, 'r'))
    re_aid_map = pickle.load(open(re_aid_map_file, 'r'))
    return re_uid_map, re_aid_map, pd.read_csv(test_file)



def predict(model_path, formal):
    user_data_file = os.path.join(model_path, 'user_data_file.csv')
    ad_data_file = os.path.join(model_path, 'ad_data_file.csv')
    graph_hyper_params_file = os.path.join(model_path, 'graph_hyper_params_file.pic')

    user_data = pd.read_csv(user_data_file)
    ad_data = pd.read_csv(ad_data_file)
    re_uid_map, re_aid_map, ori_test = get_map_dict_and_predict(formal)
    graph_hyper_params = pickle.load(open(graph_hyper_params_file, 'r'))

    uid_map = dict(zip(re_uid_map.values(), re_uid_map.keys()))
    aid_map = dict(zip(re_aid_map.values(), re_aid_map.keys()))
    # ori_test['uid'].map(lambda x: uid_map[x])
    # ori_test['aid'].map(lambda x: aid_map[x])
    ori_test['aid'] = ori_test['aid'].apply(lambda x: aid_map[x])
    ori_test['uid'] = ori_test['uid'].apply(lambda x: uid_map[x])
    predict_data = ori_test


    checkpoint_file = tf.train.latest_checkpoint(model_path)
    split_vector_data = SplitClass()
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            print "{}.meta".format(checkpoint_file)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # print graph.get_operations()

            uid_p = graph.get_operation_by_name("uid").outputs[0]
            lbs_p = graph.get_operation_by_name("LBS").outputs[0]
            age_p = graph.get_operation_by_name("age").outputs[0]

            carrier_p = graph.get_operation_by_name("carrier").outputs[0]
            consumptionability_p = graph.get_operation_by_name("consumptionAbility").outputs[0]
            education_p = graph.get_operation_by_name("education").outputs[0]
            gender_p = graph.get_operation_by_name("gender").outputs[0]
            house_p = graph.get_operation_by_name("house").outputs[0]
            os_p = graph.get_operation_by_name("os").outputs[0]
            ct_p = graph.get_operation_by_name("ct").outputs[0]
            marriagestatus_p = graph.get_operation_by_name("marriageStatus").outputs[0]

            appidaction_index_p = graph.get_operation_by_name("appidaction_index").outputs[0]
            appidaction_val_p = graph.get_operation_by_name("appidaction_val").outputs[0]
            appIdInstall_index_p = graph.get_operation_by_name("appIdInstall_index").outputs[0]
            appIdInstall_val_p = graph.get_operation_by_name("appIdInstall_val").outputs[0]

            interest1_index_p = graph.get_operation_by_name("interest1_index").outputs[0]
            interest1_val_p = graph.get_operation_by_name("interest1_val").outputs[0]
            interest2_index_p = graph.get_operation_by_name("interest2_index").outputs[0]
            interest2_val_p = graph.get_operation_by_name("interest2_val").outputs[0]
            interest3_index_p = graph.get_operation_by_name("interest3_index").outputs[0]
            interest3_val_p = graph.get_operation_by_name("interest3_val").outputs[0]
            interest4_index_p = graph.get_operation_by_name("interest4_index").outputs[0]
            interest4_val_p = graph.get_operation_by_name("interest4_val").outputs[0]
            interest5_index_p = graph.get_operation_by_name("interest5_index").outputs[0]
            interest5_val_p = graph.get_operation_by_name("interest5_val").outputs[0]

            kw1_index_p = graph.get_operation_by_name("kw1_index").outputs[0]
            kw1_val_p = graph.get_operation_by_name("kw1_val").outputs[0]
            kw2_index_p = graph.get_operation_by_name("kw2_index").outputs[0]
            kw2_val_p = graph.get_operation_by_name("kw2_val").outputs[0]
            kw3_index_p = graph.get_operation_by_name("kw3_index").outputs[0]
            kw3_val_p = graph.get_operation_by_name("kw3_val").outputs[0]

            topic1_index_p = graph.get_operation_by_name("topic1_index").outputs[0]
            topic1_val_p = graph.get_operation_by_name("topic1_val").outputs[0]
            topic2_index_p = graph.get_operation_by_name("topic2_index").outputs[0]
            topic2_val_p = graph.get_operation_by_name("topic2_val").outputs[0]
            topic3_index_p = graph.get_operation_by_name("topic3_index").outputs[0]
            topic3_val_p = graph.get_operation_by_name("topic3_val").outputs[0]

            aid_p = graph.get_operation_by_name("aid").outputs[0]
            advertiserid_p = graph.get_operation_by_name("advertiserId").outputs[0]
            campaignid_p = graph.get_operation_by_name("campaignId").outputs[0]
            creativeid_p = graph.get_operation_by_name("creativeId").outputs[0]
            adcategoryid_p = graph.get_operation_by_name("adCategoryId").outputs[0]
            productid_p = graph.get_operation_by_name("productId").outputs[0]
            producttype_p = graph.get_operation_by_name("productType").outputs[0]
            creativesize_p = graph.get_operation_by_name("creativeSize").outputs[0]

            train_p = graph.get_operation_by_name("train_p").outputs[0]
            dropout_p = graph.get_operation_by_name("dropout_p").outputs[0]

            pred_val = graph.get_operation_by_name("OutAll/final/pred").outputs[0]

            print 'map row start'
            uid_map_row, aid_map_row = dict(
                zip(user_data['uid'].values, np.arange(len(user_data)))), dict(
                zip(ad_data['aid'].values, np.arange(len(ad_data))))
            print 'map row end'


            def get_fed_dict(formal, uid_map_row, aid_map_row, pda, spv):
                if formal:
                    aid_list = pda['aid'].values
                    uid_list = pda['uid'].values
                else:
                    if len(pda) == 4:
                        aid_list, uid_list = [11, 11, 11, 11], [11, 190, 191, 11]
                    elif len(pda) == 3:
                        aid_list, uid_list = [11, 11, 11], [11, 190, 191]
                    else:
                        aid_list, uid_list = [11], [11]

                # print 11
                # d1 = datetime.now()
                b_u_d, b_a_d = [], []
                for b_uid in uid_list:
                    b_u_d.append(user_data.iloc[uid_map_row[b_uid]])
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
                appidaction_li = spv(b_u_d['appIdAction'])
                # print 1212
                fed_dict[appidaction_index_p], fed_dict[appidaction_val_p] = appidaction_li[0], appidaction_li[1]
                appIdInstall_li = spv(b_u_d['appIdInstall'])
                fed_dict[appIdInstall_index_p], fed_dict[appIdInstall_val_p] = appIdInstall_li[0], appIdInstall_li[1]
                # print 122
                interest1_li = spv(b_u_d['interest1'])
                fed_dict[interest1_index_p], fed_dict[interest1_val_p] = interest1_li[0], interest1_li[1]
                interest2_li = spv(b_u_d['interest2'])
                fed_dict[interest2_index_p], fed_dict[interest2_val_p] = interest2_li[0], interest2_li[1]
                interest3_li = spv(b_u_d['interest3'])
                fed_dict[interest3_index_p], fed_dict[interest3_val_p] = interest3_li[0], interest3_li[1]
                interest4_li = spv(b_u_d['interest4'])
                fed_dict[interest4_index_p], fed_dict[interest4_val_p] = interest4_li[0], interest4_li[1]
                interest5_li = spv(b_u_d['interest5'])
                fed_dict[interest5_index_p], fed_dict[interest5_val_p] = interest5_li[0], interest5_li[1]
                # print 123
                kw1_li = spv(b_u_d['kw1'])
                fed_dict[kw1_index_p], fed_dict[kw1_val_p] = kw1_li[0], kw1_li[1]
                kw2_li = spv(b_u_d['kw2'])
                fed_dict[kw2_index_p], fed_dict[kw2_val_p] = kw2_li[0], kw2_li[1]
                kw3_li = spv(b_u_d['kw3'])
                fed_dict[kw3_index_p], fed_dict[kw3_val_p] = kw3_li[0], kw3_li[1]
                # print 124
                topic1_li = spv(b_u_d['topic1'])
                fed_dict[topic1_index_p], fed_dict[topic1_val_p] = topic1_li[0], topic1_li[1]
                topic2_li = spv(b_u_d['topic2'])
                fed_dict[topic2_index_p], fed_dict[topic2_val_p] = topic2_li[0], topic2_li[1]
                topic3_li = spv(b_u_d['topic3'])
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
                if graph_hyper_params['creativeSize_pro'] == 'min_max':
                    fed_dict[creativesize_p] = np.expand_dims(b_a_d['creativeSize'], axis=1).astype(np.float32)
                elif graph_hyper_params['creativeSize_pro'] == 'li_san':
                    fed_dict[creativesize_p] = np.expand_dims(b_a_d['creativeSize'], axis=1)
                else:
                    print 'wrong feed'
                return fed_dict


            e_b_s = len(predict_data) / graph_hyper_params['batch_size'] if len(predict_data) % graph_hyper_params[
                'batch_size'] == 0 else len(predict_data) / graph_hyper_params['batch_size'] + 1
            pred = []
            for index in tqdm(range(e_b_s)):
                start = index * graph_hyper_params['batch_size']
                end = (index + 1) * graph_hyper_params['batch_size'] if (index + 1) * graph_hyper_params['batch_size'] < len(predict_data) else len(predict_data) + 1
                b_predict_data = predict_data[start:end]
                # print len(b_predict_data), start, end
                fed_dict = get_fed_dict(formal, uid_map_row, aid_map_row, b_predict_data, split_vector_data)
                fed_dict[train_p] = False
                fed_dict[dropout_p] = np.array([1.0])
                pred_value = sess.run([pred_val], feed_dict=fed_dict)
                # print pred_value
                pre_real_val = np.array(pred_value).reshape((-1))
                pred = pred + pre_real_val.tolist()


            # print len(pred), len(predict_data)
            predict_data['pred_label'] = pred
            # print re_aid_map
            predict_data['ori_aid'] = predict_data['aid'].apply(lambda x: re_aid_map[x])
            predict_data['ori_uid'] = predict_data['uid'].apply(lambda x: re_uid_map[x])
            csv_data = predict_data[['ori_aid', 'ori_uid', 'pred_label']]
            csv_data.columns = ['aid', 'uid', 'score']
            csv_path = os.path.join(model_path,'submission.csv')
            csv_data.to_csv(csv_path, index=False)
            print 'submission_path:', csv_path
            return csv_path
    pass


def parse():
    args = argparse.ArgumentParser(description='Ten Con !')
    args.add_argument('--mp', type=str, default='none', help='model path')
    args.add_argument('--fm', type=bool, default=False, help='formal')
    return args


if __name__ == '__main__':
    args = parse()
    args = args.parse_args()

    print args.mp
    print args.fm

    # predict('/Users/Xuehj/Desktop/TencentContest/CodeBase/stage3_dmf_tencent/checkpoints/dmf_tencent/2018-05-03-19-58-40', False)
    # predict('/Users/Xuehj/Desktop/TencentContest/CodeBase/dmf_tencent/checkpoints/dmf_tencent/2018-04-30-10-35-22/', False)
    predict(args.mp, args.fm)
    pass