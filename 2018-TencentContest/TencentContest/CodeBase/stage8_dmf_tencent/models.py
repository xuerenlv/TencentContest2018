# coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


# one_hot_feature = ['aid', 'uid', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education',
#                    'gender', 'house', 'os', 'ct', 'marriageStatus', 'advertiserId',
#                    'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']
# vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2',
#                   'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
# continuous_feature = 'creativeSize'

# user_features = ['uid', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education',
#                  'gender', 'house', 'os', 'ct', 'marriageStatus',
#                  'appIdAction', 'appIdInstall', 'interest1', 'interest2',
#                  'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
# ad_features = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType',
#                'creativeSize']

def batch_norm_layer(x, train_p, batch_norm_decay, scope_bn):
    bn_train = batch_norm(x, decay=batch_norm_decay, center=True, scale=True, updates_collections=None,
                          is_training=True, reuse=None, trainable=True, scope=scope_bn)
    bn_inference = batch_norm(x, decay=batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=False, reuse=True, trainable=True, scope=scope_bn)
    z = tf.cond(train_p, lambda: bn_train, lambda: bn_inference)
    return z


# def dropout_layer(x, is_train, drp):
#     return tf.nn.dropout(x, drp) if is_train else tf.nn.dropout(x, 1.0)


def inference(uid_p, lbs_p, age_p, carrier_p, consumptionability_p, education_p, gender_p, house_p,
    os_p, ct_p, marriagestatus_p, appidaction_index_p, appidaction_val_p, appIdInstall_index_p,
    appIdInstall_val_p , interest1_index_p, interest1_val_p, interest2_index_p, interest2_val_p,
    interest3_index_p, interest3_val_p , interest4_index_p, interest4_val_p, interest5_index_p,
    interest5_val_p, kw1_index_p, kw1_val_p, kw2_index_p, kw2_val_p, kw3_index_p, kw3_val_p, topic1_index_p,
    topic1_val_p, topic2_index_p, topic2_val_p, topic3_index_p, topic3_val_p, aid_p, advertiserid_p, campaignid_p,
    creativeid_p,adcategoryid_p, productid_p, producttype_p, creativesize_p, true_label, feature_conf_dict, graph_hyper_params,
              train_p, dropout_p, user_feature_start, ad_feature_start, user_input_len, user_all_len, ad_all_len):
    regularizer = tf.contrib.layers.l2_regularizer(graph_hyper_params['l2_reg_alpha'])


    if graph_hyper_params['model'] == 'dmf':
        # emb_size, low_emb_size = 300, 150
        emb_size, low_emb_size = 32, 32
        ad_emb_size = 10
    elif 'fm' in graph_hyper_params['model']:
        emb_size, low_emb_size = 150, 150
    else:
        emb_size, low_emb_size = 0, 0
        print 'no this model infer !'

    cross_emb_size = 30
    print emb_size, low_emb_size, cross_emb_size
    with tf.variable_scope("OutAll"):
        # uid_emb = tf.get_variable("uid_emb", shape=(feature_conf_dict['uid'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        LBS_emb = tf.get_variable("LBS_emb", shape=(feature_conf_dict['LBS'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        age_emb = tf.get_variable("age_emb", shape=(feature_conf_dict['age'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        carrier_emb = tf.get_variable("carrier_emb", shape=(feature_conf_dict['carrier'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        consumptionAbility_emb = tf.get_variable("consumptionAbility_emb", shape=(feature_conf_dict['consumptionAbility'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        education_emb = tf.get_variable("education_emb", shape=(feature_conf_dict['education'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        gender_emb = tf.get_variable("gender_emb", shape=(feature_conf_dict['gender'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        house_emb = tf.get_variable("house_emb", shape=(feature_conf_dict['house'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        os_emb = tf.get_variable("os_emb", shape=(feature_conf_dict['os'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        ct_emb = tf.get_variable("ct_emb", shape=(feature_conf_dict['ct'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        marriageStatus_emb = tf.get_variable("marriageStatus_emb", shape=(feature_conf_dict['marriageStatus'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        appIdAction_emb = tf.get_variable("appIdAction_emb", shape=(feature_conf_dict['appIdAction'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        appIdInstall_emb = tf.get_variable("appIdInstall_emb", shape=(feature_conf_dict['appIdInstall'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest1_emb = tf.get_variable("interest1_emb", shape=(feature_conf_dict['interest1'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest2_emb = tf.get_variable("interest2_emb", shape=(feature_conf_dict['interest2'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest3_emb = tf.get_variable("interest3_emb", shape=(feature_conf_dict['interest3'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest4_emb = tf.get_variable("interest4_emb", shape=(feature_conf_dict['interest4'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,regularizer=regularizer)
        interest5_emb = tf.get_variable("interest5_emb", shape=(feature_conf_dict['interest5'][0], emb_size),initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        kw1_emb = tf.get_variable("kw1_emb", shape=(feature_conf_dict['kw1'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        kw2_emb = tf.get_variable("kw2_emb", shape=(feature_conf_dict['kw2'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        kw3_emb = tf.get_variable("kw3_emb", shape=(feature_conf_dict['kw3'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        topic1_emb = tf.get_variable("topic1_emb", shape=(feature_conf_dict['topic1'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        topic2_emb = tf.get_variable("topic2_emb", shape=(feature_conf_dict['topic2'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        topic3_emb = tf.get_variable("topic3_emb", shape=(feature_conf_dict['topic3'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        # aid_emb = tf.get_variable("aid_emb", shape=(feature_conf_dict['aid'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        advertiserId_emb = tf.get_variable("advertiserId_emb", shape=(feature_conf_dict['advertiserId'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        campaignId_emb = tf.get_variable("campaignId_emb", shape=(feature_conf_dict['campaignId'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        creativeId_emb = tf.get_variable("creativeId_emb", shape=(feature_conf_dict['creativeId'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        adCategoryId_emb = tf.get_variable("adCategoryId_emb", shape=(feature_conf_dict['adCategoryId'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        productId_emb = tf.get_variable("productId_emb", shape=(feature_conf_dict['productId'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        productType_emb = tf.get_variable("productType_emb", shape=(feature_conf_dict['productType'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        # bias
        # uid_emb_bias = tf.get_variable("uid_emb_bias", shape=(feature_conf_dict['uid'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        LBS_emb_bias = tf.get_variable("LBS_emb_bias", shape=(feature_conf_dict['LBS'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,regularizer=regularizer)
        age_emb_bias = tf.get_variable("age_emb_bias", shape=(feature_conf_dict['age'], 1),initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        carrier_emb_bias = tf.get_variable("carrier_emb_bias", shape=(feature_conf_dict['carrier'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        consumptionAbility_emb_bias = tf.get_variable("consumptionAbility_emb_bias", shape=(feature_conf_dict['consumptionAbility'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        education_emb_bias = tf.get_variable("education_emb_bias", shape=(feature_conf_dict['education'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        gender_emb_bias = tf.get_variable("gender_emb_bias", shape=(feature_conf_dict['gender'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        house_emb_bias = tf.get_variable("house_emb_bias", shape=(feature_conf_dict['house'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        os_emb_bias = tf.get_variable("os_emb_bias", shape=(feature_conf_dict['os'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        ct_emb_bias = tf.get_variable("ct_emb_bias", shape=(feature_conf_dict['ct'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        marriageStatus_emb_bias = tf.get_variable("marriageStatus_emb_bias", shape=(feature_conf_dict['marriageStatus'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        appIdAction_emb_bias = tf.get_variable("appIdAction_emb_bias", shape=(feature_conf_dict['appIdAction'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        appIdInstall_emb_bias = tf.get_variable("appIdInstall_emb_bias", shape=(feature_conf_dict['appIdInstall'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest1_emb_bias = tf.get_variable("interest1_emb_bias", shape=(feature_conf_dict['interest1'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest2_emb_bias = tf.get_variable("interest2_emb_bias", shape=(feature_conf_dict['interest2'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest3_emb_bias = tf.get_variable("interest3_emb_bias", shape=(feature_conf_dict['interest3'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest4_emb_bias = tf.get_variable("interest4_emb_bias", shape=(feature_conf_dict['interest4'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest5_emb_bias = tf.get_variable("interest5_emb_bias", shape=(feature_conf_dict['interest5'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        kw1_emb_bias = tf.get_variable("kw1_emb_bias", shape=(feature_conf_dict['kw1'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        kw2_emb_bias = tf.get_variable("kw2_emb_bias", shape=(feature_conf_dict['kw2'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        kw3_emb_bias = tf.get_variable("kw3_emb_bias", shape=(feature_conf_dict['kw3'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        topic1_emb_bias = tf.get_variable("topic1_emb_bias", shape=(feature_conf_dict['topic1'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        topic2_emb_bias = tf.get_variable("topic2_emb_bias", shape=(feature_conf_dict['topic2'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        topic3_emb_bias = tf.get_variable("topic3_emb_bias", shape=(feature_conf_dict['topic3'][0], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        # aid_emb_bias = tf.get_variable("aid_emb_bias", shape=(feature_conf_dict['aid'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        advertiserId_emb_bias = tf.get_variable("advertiserId_emb_bias", shape=(feature_conf_dict['advertiserId'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        campaignId_emb_bias = tf.get_variable("campaignId_emb_bias", shape=(feature_conf_dict['campaignId'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        creativeId_emb_bias = tf.get_variable("creativeId_emb_bias", shape=(feature_conf_dict['creativeId'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        adCategoryId_emb_bias = tf.get_variable("adCategoryId_emb_bias", shape=(feature_conf_dict['adCategoryId'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        productId_emb_bias = tf.get_variable("productId_emb_bias", shape=(feature_conf_dict['productId'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        productType_emb_bias = tf.get_variable("productType_emb_bias", shape=(feature_conf_dict['productType'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        # for cross module
        cross_module_emb_user = tf.get_variable("cross_len_for_emb_user", shape=(user_all_len, cross_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        cross_module_emb_ad = tf.get_variable("cross_len_for_emb_ad", shape=(ad_all_len, cross_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        # user_features = ['uid', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education',
        #                  'gender', 'house', 'os', 'ct', 'marriageStatus',
        #                  'appIdAction', 'appIdInstall', 'interest1', 'interest2',
        #                  'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
        # for cross module -- user
        user_index = tf.concat([
            lbs_p + user_feature_start['LBS'],
            age_p + user_feature_start['age'],
            carrier_p + user_feature_start['carrier'],
            consumptionability_p + user_feature_start['consumptionAbility'],
            education_p + user_feature_start['education'],
            gender_p + user_feature_start['gender'],
            house_p + user_feature_start['house'],
            os_p + user_feature_start['os'],
            ct_p + user_feature_start['ct'],
            marriagestatus_p + user_feature_start['marriageStatus'],
            appidaction_index_p + user_feature_start['appIdAction'],
            appIdInstall_index_p + user_feature_start['appIdInstall'],
            interest1_index_p + user_feature_start['interest1'],
            interest2_index_p + user_feature_start['interest2'],
            interest3_index_p + user_feature_start['interest3'],
            interest4_index_p + user_feature_start['interest4'],
            interest5_index_p + user_feature_start['interest5'],
            kw1_index_p + user_feature_start['kw1'],
            kw2_index_p + user_feature_start['kw2'],
            kw3_index_p + user_feature_start['kw3'],
            topic1_index_p + user_feature_start['topic1'],
            topic2_index_p + user_feature_start['topic2'],
            topic3_index_p + user_feature_start['topic3'],
        ], axis=-1)

        ones_tmp = tf.ones_like(lbs_p, dtype=tf.int32)
        user_index_features = tf.concat([
            ones_tmp - 1, ones_tmp, ones_tmp + 1, ones_tmp + 2, ones_tmp + 3,ones_tmp + 4,
            ones_tmp + 5, ones_tmp + 6, ones_tmp + 7, ones_tmp + 8,
            tf.tile(ones_tmp + 9, [1, feature_conf_dict['appIdAction'][1]]),
            tf.tile(ones_tmp + 10, [1, feature_conf_dict['appIdInstall'][1]]),
            tf.tile(ones_tmp + 11, [1, feature_conf_dict['interest1'][0]]),
            tf.tile(ones_tmp + 12, [1, feature_conf_dict['interest2'][0]]),
            tf.tile(ones_tmp + 13, [1, feature_conf_dict['interest3'][0]]),
            tf.tile(ones_tmp + 14, [1, feature_conf_dict['interest4'][0]]),
            tf.tile(ones_tmp + 15, [1, feature_conf_dict['interest5'][0]]),
            tf.tile(ones_tmp + 16, [1, feature_conf_dict['kw1'][1]]),
            tf.tile(ones_tmp + 17, [1, feature_conf_dict['kw2'][1]]),
            tf.tile(ones_tmp + 18, [1, feature_conf_dict['kw3'][1]]),
            tf.tile(ones_tmp + 19, [1, feature_conf_dict['topic1'][1]]),
            tf.tile(ones_tmp + 20, [1, feature_conf_dict['topic2'][1]]),
            tf.tile(ones_tmp + 21, [1, feature_conf_dict['topic3'][1]]),
        ], axis=-1)
        user_index_features_emb = tf.get_variable("user_index_features_emb", shape=[23, cross_emb_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        user_index_features_vec = tf.nn.embedding_lookup(user_index_features_emb, tf.tile(user_index_features, [1, 7]))


        user_val = tf.concat([
            tf.tile(tf.expand_dims(tf.ones_like(lbs_p, dtype=tf.float32), -1), [1, 1, 10]),
            appidaction_val_p, appIdInstall_val_p, interest1_val_p, interest2_val_p, interest3_val_p,
            interest4_val_p, interest5_val_p, kw1_val_p, kw2_val_p, kw3_val_p, topic1_val_p, topic2_val_p, topic3_val_p,
        ], axis=-1)

        # ad_features_for_cross = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId',
        #                          'productType',
        #                          'creativeSize_cross']
        # user_input_len
        ad_cross_index = tf.concat([
            tf.tile(advertiserid_p + ad_feature_start['advertiserId'], [1, user_input_len]),
            tf.tile(campaignid_p + ad_feature_start['campaignId'], [1, user_input_len]),
            tf.tile(creativeid_p + ad_feature_start['creativeId'], [1, user_input_len]),
            tf.tile(adcategoryid_p + ad_feature_start['adCategoryId'], [1, user_input_len]),
            tf.tile(productid_p + ad_feature_start['productId'], [1, user_input_len]),
            tf.tile(producttype_p + ad_feature_start['productType'], [1, user_input_len]),
            tf.tile(creativesize_p + ad_feature_start['creativeSize_cross'], [1, user_input_len]),
        ], axis=-1)

        ad_index_features = tf.concat([
            tf.tile(ones_tmp - 1, [1, user_input_len]),
            tf.tile(ones_tmp, [1, user_input_len]),
            tf.tile(ones_tmp + 1, [1, user_input_len]),
            tf.tile(ones_tmp + 2, [1, user_input_len]),
            tf.tile(ones_tmp + 3, [1, user_input_len]),
            tf.tile(ones_tmp + 4, [1, user_input_len]),
            tf.tile(ones_tmp + 5, [1, user_input_len]),
            ], axis=-1)
        ad_index_features_emb = tf.get_variable("ad_index_features_emb", shape=[7, cross_emb_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        ad_index_features_vec = tf.nn.embedding_lookup(ad_index_features_emb, ad_index_features)

        user_v_ori = tf.nn.embedding_lookup(cross_module_emb_user, tf.tile(user_index, [1, 7]))
        user_v_ori = tf.concat([user_v_ori, user_index_features_vec], axis=-1)

        ad_v_ori = tf.nn.embedding_lookup(cross_module_emb_ad, ad_cross_index)
        ad_v_ori = tf.concat([ad_v_ori, ad_index_features_vec], axis=-1)

        # cross_emb_ori = tf.concat([user_v_ori, ad_v_ori], axis=-1)
        cross_emb_ori = user_v_ori * ad_v_ori
        # cross_emb_ori = user_v_ori * ad_v_ori
        all_cross_module_val = tf.tile(user_val, [1, 1, 7])

        cross_b1 = tf.get_variable("cross_b1", shape=[cross_emb_size*2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        cross_net1 = tf.nn.dropout(tf.nn.relu(tf.reshape(tf.matmul(all_cross_module_val, cross_emb_ori), [-1, cross_emb_size*2]) + cross_b1), dropout_p[0])
        # cross_net1 = tf.nn.relu(tf.reshape(tf.matmul(all_cross_module_val, cross_emb_ori), [-1, cross_emb_size*2]))

        wwf = tf.get_variable("wwf", shape=(cross_emb_size*2, 2), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        wbf = tf.get_variable("wbf", shape=[2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        wide_pred = tf.matmul(cross_net1, wwf) + wbf



        # user
        lbs_v_ori = tf.nn.embedding_lookup(LBS_emb, lbs_p)
        lbs_v_ori_bias = tf.nn.embedding_lookup(LBS_emb_bias, lbs_p)
        lbs_v = tf.reshape(lbs_v_ori, [-1, emb_size])

        age_v_ori = tf.nn.embedding_lookup(age_emb, age_p)
        age_v_ori_bias = tf.nn.embedding_lookup(age_emb_bias, age_p)
        age_v = tf.reshape(age_v_ori, [-1, emb_size])

        carrier_v_ori = tf.nn.embedding_lookup(carrier_emb, carrier_p)
        carrier_v_ori_bias = tf.nn.embedding_lookup(carrier_emb_bias, carrier_p)
        carrier_v = tf.reshape(carrier_v_ori, [-1, emb_size])

        consumptionability_v_ori = tf.nn.embedding_lookup(consumptionAbility_emb, consumptionability_p)
        consumptionability_v_ori_bias = tf.nn.embedding_lookup(consumptionAbility_emb_bias, consumptionability_p)
        consumptionability_v = tf.reshape(consumptionability_v_ori, [-1, emb_size])

        education_v_ori = tf.nn.embedding_lookup(education_emb, education_p)
        education_v_ori_bias = tf.nn.embedding_lookup(education_emb_bias, education_p)
        education_v = tf.reshape(education_v_ori, [-1, emb_size])

        gender_v_ori = tf.nn.embedding_lookup(gender_emb, gender_p)
        gender_v_ori_bias = tf.nn.embedding_lookup(gender_emb_bias, gender_p)
        gender_v = tf.reshape(gender_v_ori, [-1, emb_size])

        house_v_ori = tf.nn.embedding_lookup(house_emb, house_p)
        house_v_ori_bias = tf.nn.embedding_lookup(house_emb_bias, house_p)
        house_v = tf.reshape(house_v_ori, [-1, emb_size])

        os_v_ori = tf.nn.embedding_lookup(os_emb, os_p)
        os_v_ori_bias = tf.nn.embedding_lookup(os_emb_bias, os_p)
        os_v = tf.reshape(os_v_ori, [-1, emb_size])

        ct_v_ori = tf.nn.embedding_lookup(ct_emb, ct_p)
        ct_v_ori_bias = tf.nn.embedding_lookup(ct_emb_bias, ct_p)
        ct_v = tf.reshape(ct_v_ori, [-1, emb_size])

        marriagestatus_v_ori = tf.nn.embedding_lookup(marriageStatus_emb, marriagestatus_p)
        marriagestatus_v_ori_bias = tf.nn.embedding_lookup(marriageStatus_emb_bias, marriagestatus_p)
        marriagestatus_v = tf.reshape(marriagestatus_v_ori, [-1, emb_size])

        appIdAction_v_ori = tf.nn.embedding_lookup(appIdAction_emb, appidaction_index_p)
        appIdAction_v_ori_bias = tf.nn.embedding_lookup(appIdAction_emb_bias, appidaction_index_p)
        appIdAction_v_ori_prod = tf.multiply(tf.transpose(appidaction_val_p, [0, 2, 1]), appIdAction_v_ori)
        appIdAction_v_ori_prod_bias = tf.multiply(tf.transpose(appidaction_val_p, [0, 2, 1]), appIdAction_v_ori_bias)
        appIdAction_v = tf.reshape(tf.matmul(appidaction_val_p, appIdAction_v_ori), [-1, emb_size])

        appIdInstall_v_ori = tf.nn.embedding_lookup(appIdInstall_emb, appIdInstall_index_p)
        appIdInstall_v_ori_bias = tf.nn.embedding_lookup(appIdInstall_emb_bias, appIdInstall_index_p)
        appIdInstall_v_ori_prod = tf.multiply(tf.transpose(appIdInstall_val_p, [0, 2, 1]), appIdInstall_v_ori)
        appIdInstall_v_ori_prod_bias = tf.multiply(tf.transpose(appIdInstall_val_p, [0, 2, 1]), appIdInstall_v_ori_bias)
        appIdInstall_v = tf.reshape(tf.matmul(appIdInstall_val_p, appIdInstall_v_ori), [-1, low_emb_size])

        interest1_v_ori = tf.nn.embedding_lookup(interest1_emb, interest1_index_p)
        interest1_v_ori_bias = tf.nn.embedding_lookup(interest1_emb_bias, interest1_index_p)
        interest1_v_ori_prod = tf.multiply(tf.transpose(interest1_val_p, [0, 2, 1]), interest1_v_ori)
        interest1_v_ori_prod_bias = tf.multiply(tf.transpose(interest1_val_p, [0, 2, 1]), interest1_v_ori_bias)
        interest1_v = tf.reshape(tf.matmul(interest1_val_p, interest1_v_ori), [-1, emb_size])
        interest1_v_det = tf.reshape(interest1_v_ori_prod, [-1, feature_conf_dict['interest1'][0]*emb_size])

        interest2_v_ori = tf.nn.embedding_lookup(interest2_emb, interest2_index_p)
        interest2_v_ori_bias = tf.nn.embedding_lookup(interest2_emb_bias, interest2_index_p)
        interest2_v_ori_prod = tf.multiply(tf.transpose(interest2_val_p, [0, 2, 1]), interest2_v_ori)
        interest2_v_ori_prod_bias = tf.multiply(tf.transpose(interest2_val_p, [0, 2, 1]), interest2_v_ori_bias)
        interest2_v = tf.reshape(tf.matmul(interest2_val_p, interest2_v_ori), [-1, emb_size])
        interest2_v_det = tf.reshape(interest2_v_ori_prod, [-1, feature_conf_dict['interest2'][0] * emb_size])

        interest3_v_ori = tf.nn.embedding_lookup(interest3_emb, interest3_index_p)
        interest3_v_ori_bias = tf.nn.embedding_lookup(interest3_emb_bias, interest3_index_p)
        interest3_v_ori_prod = tf.multiply(tf.transpose(interest3_val_p, [0, 2, 1]), interest3_v_ori)
        interest3_v_ori_prod_bias = tf.multiply(tf.transpose(interest3_val_p, [0, 2, 1]), interest3_v_ori_bias)
        interest3_v = tf.reshape(tf.matmul(interest3_val_p, interest3_v_ori), [-1, emb_size])
        interest3_v_det = tf.reshape(interest3_v_ori_prod, [-1, feature_conf_dict['interest3'][0] * emb_size])

        interest4_v_ori = tf.nn.embedding_lookup(interest4_emb, interest4_index_p)
        interest4_v_ori_bias = tf.nn.embedding_lookup(interest4_emb_bias, interest4_index_p)
        interest4_v_ori_prod = tf.multiply(tf.transpose(interest4_val_p, [0, 2, 1]), interest4_v_ori)
        interest4_v_ori_prod_bias = tf.multiply(tf.transpose(interest4_val_p, [0, 2, 1]), interest4_v_ori_bias)
        interest4_v = tf.reshape(tf.matmul(interest4_val_p, interest4_v_ori), [-1, emb_size])
        interest4_v_det = tf.reshape(interest4_v_ori_prod, [-1, feature_conf_dict['interest4'][0] * emb_size])

        interest5_v_ori = tf.nn.embedding_lookup(interest5_emb, interest5_index_p)
        interest5_v_ori_bias = tf.nn.embedding_lookup(interest5_emb_bias, interest5_index_p)
        interest5_v_ori_prod = tf.multiply(tf.transpose(interest5_val_p, [0, 2, 1]), interest5_v_ori)
        interest5_v_ori_prod_bias = tf.multiply(tf.transpose(interest5_val_p, [0, 2, 1]), interest5_v_ori_bias)
        interest5_v = tf.reshape(tf.matmul(interest5_val_p, interest5_v_ori), [-1, emb_size])
        interest5_v_det = tf.reshape(interest5_v_ori_prod, [-1, feature_conf_dict['interest5'][0] * emb_size])

        kw1_v_ori= tf.nn.embedding_lookup(kw1_emb, kw1_index_p)
        kw1_v_ori_bias= tf.nn.embedding_lookup(kw1_emb_bias, kw1_index_p)
        kw1_v_ori_prod = tf.multiply(tf.transpose(kw1_val_p, [0, 2, 1]), kw1_v_ori)
        kw1_v_ori_prod_bias = tf.multiply(tf.transpose(kw1_val_p, [0, 2, 1]), kw1_v_ori_bias)
        kw1_v = tf.reshape(tf.matmul(kw1_val_p, kw1_v_ori), [-1, low_emb_size])

        kw2_v_ori = tf.nn.embedding_lookup(kw2_emb, kw2_index_p)
        kw2_v_ori_bias = tf.nn.embedding_lookup(kw2_emb_bias, kw2_index_p)
        kw2_v_ori_prod = tf.multiply(tf.transpose(kw2_val_p, [0, 2, 1]), kw2_v_ori)
        kw2_v_ori_prod_bias = tf.multiply(tf.transpose(kw2_val_p, [0, 2, 1]), kw2_v_ori_bias)
        kw2_v = tf.reshape(tf.matmul(kw2_val_p, kw2_v_ori), [-1, low_emb_size])

        kw3_v_ori = tf.nn.embedding_lookup(kw3_emb, kw3_index_p)
        kw3_v_ori_bias = tf.nn.embedding_lookup(kw3_emb_bias, kw3_index_p)
        kw3_v_ori_prod = tf.multiply(tf.transpose(kw3_val_p, [0, 2, 1]), kw3_v_ori)
        kw3_v_ori_prod_bias = tf.multiply(tf.transpose(kw3_val_p, [0, 2, 1]), kw3_v_ori_bias)
        kw3_v = tf.reshape(tf.matmul(kw3_val_p, kw3_v_ori), [-1, low_emb_size])

        topic1_v_ori = tf.nn.embedding_lookup(topic1_emb, topic1_index_p)
        topic1_v_ori_bias = tf.nn.embedding_lookup(topic1_emb_bias, topic1_index_p)
        topic1_v_ori_prod = tf.multiply(tf.transpose(topic1_val_p, [0, 2, 1]), topic1_v_ori)
        topic1_v_ori_prod_bias = tf.multiply(tf.transpose(topic1_val_p, [0, 2, 1]), topic1_v_ori_bias)
        topic1_v = tf.reshape(tf.matmul(topic1_val_p, topic1_v_ori), [-1, emb_size])

        topic2_v_ori = tf.nn.embedding_lookup(topic2_emb, topic2_index_p)
        topic2_v_ori_bias = tf.nn.embedding_lookup(topic2_emb_bias, topic2_index_p)
        topic2_v_ori_prod = tf.multiply(tf.transpose(topic2_val_p, [0, 2, 1]), topic2_v_ori)
        topic2_v_ori_prod_bias = tf.multiply(tf.transpose(topic2_val_p, [0, 2, 1]), topic2_v_ori_bias)
        topic2_v = tf.reshape(tf.matmul(topic2_val_p, topic2_v_ori), [-1, emb_size])

        topic3_v_ori = tf.nn.embedding_lookup(topic3_emb, topic3_index_p)
        topic3_v_ori_bias = tf.nn.embedding_lookup(topic3_emb_bias, topic3_index_p)
        topic3_v_ori_prod = tf.multiply(tf.transpose(topic3_val_p, [0, 2, 1]), topic3_v_ori)
        topic3_v_ori_prod_bias = tf.multiply(tf.transpose(topic3_val_p, [0, 2, 1]), topic3_v_ori_bias)
        topic3_v = tf.reshape(tf.matmul(topic3_val_p, topic3_v_ori), [-1, emb_size])

        # ad
        advertiserid_v_ori = tf.nn.embedding_lookup(advertiserId_emb, advertiserid_p)
        advertiserid_v_ori_bias = tf.nn.embedding_lookup(advertiserId_emb_bias, advertiserid_p)
        advertiserid_v = tf.reshape(advertiserid_v_ori, [-1, ad_emb_size])

        campaignid_v_ori = tf.nn.embedding_lookup(campaignId_emb, campaignid_p)
        campaignid_v_ori_bias = tf.nn.embedding_lookup(campaignId_emb_bias, campaignid_p)
        campaignid_v = tf.reshape(campaignid_v_ori, [-1, ad_emb_size])

        creativeid_v_ori = tf.nn.embedding_lookup(creativeId_emb, creativeid_p)
        creativeid_v_ori_bias = tf.nn.embedding_lookup(creativeId_emb_bias, creativeid_p)
        creativeid_v = tf.reshape(creativeid_v_ori, [-1, ad_emb_size])

        adcategoryid_v_ori = tf.nn.embedding_lookup(adCategoryId_emb, adcategoryid_p)
        adcategoryid_v_ori_bias = tf.nn.embedding_lookup(adCategoryId_emb_bias, adcategoryid_p)
        adcategoryid_v = tf.reshape(adcategoryid_v_ori, [-1, ad_emb_size])

        productid_v_ori = tf.nn.embedding_lookup(productId_emb, productid_p)
        productid_v_ori_bias = tf.nn.embedding_lookup(productId_emb_bias, productid_p)
        productid_v = tf.reshape(productid_v_ori, [-1, ad_emb_size])

        producttype_v_ori = tf.nn.embedding_lookup(productType_emb, producttype_p)
        producttype_v_ori_bias = tf.nn.embedding_lookup(productType_emb_bias, producttype_p)
        producttype_v = tf.reshape(producttype_v_ori, [-1, ad_emb_size])

        creativesize_v, creativesize_v_ori, creativesize_v_ori_bias = None, None, None
        if graph_hyper_params['creativeSize_pro'] == 'min_max':
            creativesize_v = tf.reshape(creativesize_p, [-1, 1])
            ad_vector_size = ad_emb_size * 6 + 1 # for dmf
        elif graph_hyper_params['creativeSize_pro'] == 'li_san': # not for dmf
            creativesize_emb = tf.get_variable("creativesize_emb", shape=(feature_conf_dict['creativeSize'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            creativesize_emb_bias = tf.get_variable("creativesize_emb_bias", shape=(feature_conf_dict['creativeSize'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

            creativesize_v_ori = tf.nn.embedding_lookup(creativesize_emb, creativesize_p)
            creativesize_v_ori_bias = tf.nn.embedding_lookup(creativesize_emb_bias, creativesize_p)

            creativesize_v = tf.reshape(creativesize_v_ori, [-1, ad_emb_size])
            ad_vector_size = ad_emb_size * 7  # for dmf
        else:
            print 'wrong creativeSize_pro'

        # init vector
        user_vector = tf.concat([lbs_v, age_v, carrier_v, consumptionability_v, education_v, gender_v, house_v, os_v, ct_v,
                                marriagestatus_v, appIdAction_v, appIdInstall_v, interest1_v_det, interest2_v_det, interest3_v_det,
                                interest4_v_det, interest5_v_det, kw1_v, kw2_v, kw3_v, topic1_v, topic2_v, topic3_v], axis=-1)

        user_vector_size = emb_size * 14 + 4 * low_emb_size + (feature_conf_dict['interest1'][0] +
                           feature_conf_dict['interest2'][0] + feature_conf_dict['interest3'][0] + feature_conf_dict['interest4'][0]
                           +feature_conf_dict['interest5'][0]) * emb_size

        ad_vector = tf.concat([advertiserid_v, campaignid_v, creativeid_v, adcategoryid_v, productid_v, producttype_v, creativesize_v], axis=-1)

        print 'user_ad_vector_size:', user_vector_size, ad_vector_size
        if graph_hyper_params['model'] == 'dmf':
            print 'dmf model !'
            # network

            if graph_hyper_params['dmf_det']:
                print 'dmf model more detail !'
                prod_vectors_low_emb_size = [appIdInstall_v_ori_prod, kw1_v_ori_prod, kw2_v_ori_prod, kw3_v_ori_prod]
                prod_vectors_low_emb_size_name = ['appIdInstall', 'kw1', 'kw2', 'kw3']
                for i in range(len(prod_vectors_low_emb_size)):
                    prod_vectors_low_emb_size[i] = tf.reshape(prod_vectors_low_emb_size[i],
                                                              [-1, feature_conf_dict[prod_vectors_low_emb_size_name[i]][1]* low_emb_size])

                prod_vectors_emb_size = [appIdAction_v_ori_prod, interest1_v_ori_prod, interest2_v_ori_prod,
                                         interest3_v_ori_prod,interest4_v_ori_prod, interest5_v_ori_prod, topic1_v_ori_prod,
                                         topic2_v_ori_prod, topic3_v_ori_prod]
                prod_vectors_emb_size_name = ['appIdAction', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                                              'topic1', 'topic2', 'topic3']
                for i in range(len(prod_vectors_emb_size)):
                    prod_vectors_emb_size[i] = tf.reshape(prod_vectors_emb_size[i],
                                                          [-1, feature_conf_dict[prod_vectors_emb_size_name[i]][1] * emb_size])

                user_vector = tf.concat(
                    [lbs_v, age_v, carrier_v, consumptionability_v, education_v, gender_v, house_v, os_v, ct_v,
                     marriagestatus_v] + prod_vectors_low_emb_size + prod_vectors_emb_size, axis=-1)
                user_vector_size = 10 * emb_size + (feature_conf_dict['appIdInstall'][1] + feature_conf_dict['kw1'][1]
                                                    + feature_conf_dict['kw2'][1] + feature_conf_dict['kw3'][
                                                        1]) * low_emb_size + \
                                   (feature_conf_dict['appIdAction'][1] + feature_conf_dict['interest1'][1]
                                    + feature_conf_dict['interest2'][1] + feature_conf_dict['interest3'][1]
                                    + feature_conf_dict['interest4'][1] + feature_conf_dict['interest5'][1]
                                    + feature_conf_dict['topic1'][1] + feature_conf_dict['topic2'][1]
                                    + feature_conf_dict['topic3'][1]) * emb_size


            u_b1 = tf.get_variable("u_b1", shape=[user_vector_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            net_u_1_pre = user_vector + u_b1
            if graph_hyper_params['use_bn']:
                net_u_1_pre = batch_norm_layer(net_u_1_pre, train_p, 0.995, 'net_u_1_pre')
            net_u_1 = tf.nn.relu(net_u_1_pre)
            # net_u_1 = tf.nn.dropout(net_u_1, dropout_p[0])

            u_w2 = tf.get_variable("u_w2", shape=(user_vector_size, 500), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            u_b2 = tf.get_variable("u_b2", shape=[500], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            net_u_2_pre = tf.nn.relu(tf.matmul(net_u_1, u_w2) + u_b2)
            # net_u_2_pre_asf = tf.nn.dropout(tf.matmul(net_u_1, u_w2) + u_b2, dropout_p[0])
            # if graph_hyper_params['use_bn']:
            #     net_u_2_pre = batch_norm_layer(net_u_2_pre, train_p, 0.995, 'net_u_2_pre')
            # net_u_2 = tf.nn.relu(net_u_2_pre)
            # # net_u_2 = tf.nn.dropout(net_u_2, dropout_p[0])
            #
            #
            # u_w3 = tf.get_variable("u_w3", shape=(500, 500), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            # u_b3 = tf.get_variable("u_b3", shape=[500], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            # net_u_3_pre = tf.nn.dropout(tf.matmul(net_u_2, u_w3) + u_b3, dropout_p[0])
            # if graph_hyper_params['use_bn']:
            #     net_u_3_pre = batch_norm_layer(net_u_3_pre, train_p, 0.995, 'net_u_3_pre')
            # net_u_3 = tf.nn.relu(net_u_3_pre)
            # # net_u_3 = tf.nn.dropout(net_u_3_pre, dropout_p[0])
            #
            # u_w4 = tf.get_variable("u_w4", shape=(500, 500), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            # u_b4 = tf.get_variable("u_b4", shape=[500], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            # net_u_4_pre = tf.nn.dropout(tf.matmul(net_u_3, u_w4) + u_b4, dropout_p[0])
            # if graph_hyper_params['use_bn']:
            #     net_u_4_pre = batch_norm_layer(net_u_4_pre, train_p, 0.995, 'net_u_4_pre')
            # net_u_4 = tf.nn.relu(net_u_4_pre)
            # # net_u_4 = tf.nn.dropout(net_u_4_pre, dropout_p[0])

            v_b1 = tf.get_variable("v_b1", shape=[ad_vector_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            net_v_1_pre = ad_vector + v_b1
            if graph_hyper_params['use_bn']:
                net_v_1_pre = batch_norm_layer(net_v_1_pre, train_p, 0.995, 'net_v_1_pre')
            net_v_1 = tf.nn.relu(net_v_1_pre)
            # net_v_1 = tf.nn.dropout(net_v_1, dropout_p[0])

            v_w2 = tf.get_variable("v_w2", shape=(ad_vector_size, 500), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            v_b2 = tf.get_variable("v_b2", shape=[500], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            net_v_2_pre = tf.nn.relu(tf.matmul(net_v_1, v_w2) + v_b2)
            # net_v_2_pre_asf = tf.nn.dropout(tf.matmul(net_v_1, v_w2) + v_b2, dropout_p[0])
            # if graph_hyper_params['use_bn']:
            #     net_v_2_pre = batch_norm_layer(net_v_2_pre, train_p, 0.995, 'net_v_2_pre')
            # net_v_2 = tf.nn.relu(net_v_2_pre)
            # # net_v_2 = tf.nn.dropout(net_v_2, dropout_p[0])
            #
            # v_w3 = tf.get_variable("v_w3", shape=(500, 500), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            # v_b3 = tf.get_variable("v_b3", shape=[500], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            # net_v_3_pre = tf.nn.dropout(tf.matmul(net_v_2, v_w3) + v_b3, dropout_p[0])
            # if graph_hyper_params['use_bn']:
            #     net_v_3_pre = batch_norm_layer(net_v_3_pre, train_p, 0.995, 'net_v_3_pre')
            # net_v_3 = tf.nn.relu(net_v_3_pre)
            # # net_v_3 = tf.nn.dropout(net_v_3, dropout_p[0])
            #
            # v_w4 = tf.get_variable("v_w4", shape=(500, 500), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            # v_b4 = tf.get_variable("v_b4", shape=[500], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            # net_v_4_pre = tf.nn.dropout(tf.matmul(net_v_3, v_w4) + v_b4, dropout_p[0])
            # if graph_hyper_params['use_bn']:
            #     net_v_4_pre = batch_norm_layer(net_v_4_pre, train_p, 0.995, 'net_v_4_pre')
            # net_v_4 = tf.nn.relu(net_v_4_pre)
            # # net_v_3 = tf.nn.dropout(net_v_3, dropout_p[0])

            # print 'dmf 3layer res'
            # net_u_final = net_u_2_pre_asf + net_u_3_pre + net_u_4_pre + net_v_3_pre + net_v_4_pre
            # net_v_final = net_v_2_pre_asf + net_v_3_pre + net_v_4_pre + net_u_3_pre + net_u_4_pre
            # net_u_final = net_u_2_pre_asf
            # net_v_final = net_v_2_pre_asf
            net_u_final = net_u_2_pre
            net_v_final = net_v_2_pre

            if graph_hyper_params['use_kernal']:
                # norm_u = tf.sqrt(tf.reduce_sum(tf.square(net_u_final), 1, keep_dims=True))
                # norm_v = tf.sqrt(tf.reduce_sum(tf.square(net_v_final), 1, keep_dims=True))
                # fen_mu = norm_u * norm_v + 1e-6
                abs_k = tf.abs(net_u_final - net_v_final)
                inner_k = net_u_final * net_v_final


                final_vec = tf.concat([abs_k, inner_k], axis=-1)

                wf_res1 = tf.get_variable("wf_res1", shape=(1000, 1000), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                bf_res1 = tf.get_variable("bf_res1", shape=[1000], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                final_vec_wf_pre_res1 = tf.matmul(final_vec, wf_res1) + bf_res1
                final_vec_wf_res1 = tf.nn.relu(final_vec_wf_pre_res1)

                wf_res2 = tf.get_variable("wf_res2", shape=(1000, 1000), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                bf_res2 = tf.get_variable("bf_res2", shape=[1000], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                final_vec_wf_pre_res2 = tf.matmul(final_vec_wf_res1, wf_res2) + bf_res2
                final_vec_wf_res2 = tf.nn.relu(final_vec_wf_pre_res2)


                # res unit 1
                final_vec = final_vec + final_vec_wf_res2

                wf_res3 = tf.get_variable("wf_res3", shape=(1000, 1000), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,regularizer=regularizer)
                bf_res3 = tf.get_variable("bf_res3", shape=[1000], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                final_vec_wf_pre_res3 = tf.matmul(final_vec, wf_res3) + bf_res3
                final_vec_wf_res3 = tf.nn.relu(final_vec_wf_pre_res3)

                wf_res4 = tf.get_variable("wf_res4", shape=(1000, 1000), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                bf_res4 = tf.get_variable("bf_res4", shape=[1000], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                final_vec_wf_pre_res4 = tf.matmul(final_vec_wf_res3, wf_res4) + bf_res4
                final_vec_wf_res4 = tf.nn.relu(final_vec_wf_pre_res4)

                # res unit 2
                final_vec = final_vec + final_vec_wf_res4

                wf_res5 = tf.get_variable("wf_res5", shape=(1000, 1000), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                bf_res5 = tf.get_variable("bf_res5", shape=[1000], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                final_vec_wf_pre_res5 = tf.matmul(final_vec, wf_res5) + bf_res5
                final_vec_wf_res5 = tf.nn.relu(final_vec_wf_pre_res5)

                wf_res6 = tf.get_variable("wf_res6", shape=(1000, 1000), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                bf_res6 = tf.get_variable("bf_res6", shape=[1000], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                final_vec_wf_pre_res6 = tf.matmul(final_vec_wf_res5, wf_res6) + bf_res6
                final_vec_wf_res6 = tf.nn.relu(final_vec_wf_pre_res6)

                # res unit 3
                final_vec = final_vec + final_vec_wf_res6

                wf1 = tf.get_variable("wf1", shape=(1000, 128),initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                bf1 = tf.get_variable("bf1", shape=[128], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                final_vec_wf_pre_1 = tf.matmul(final_vec, wf1) + bf1
                final_vec_1 = tf.nn.dropout(tf.nn.relu(final_vec_wf_pre_1), dropout_p[0])

                # wf1_1 = tf.get_variable("wf1_1", shape=(256, 128), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                # bf1_1 = tf.get_variable("bf1_1", shape=[128], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                # final_vec_wf_pre_1_1 = tf.matmul(final_vec_1, wf1_1) + bf1_1
                # final_vec_1_1 = tf.nn.dropout(tf.nn.relu(final_vec_wf_pre_1_1), dropout_p[0])


                wf2 = tf.get_variable("wf2", shape=(128, 2), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                bf2 = tf.get_variable("bf2", shape=[2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                final_vec_wf_2 = tf.matmul(final_vec_1, wf2) + bf2
                # wide_pred

                # wf3 = tf.get_variable("wf3", shape=(500, 100), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                # bf3 = tf.get_variable("bf3", shape=[100], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                # final_vec_wf_3 = tf.nn.relu(tf.matmul(tf.nn.relu(final_vec_wf_2), wf3) + bf3)
                #
                # fffinal_vec = tf.concat([final_vec_wf, final_vec_wf_2, final_vec_wf_3], axis=-1)
                # wff = tf.get_variable("wff", shape=(1600, 2), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                # bff = tf.get_variable("bff", shape=[2], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32, regularizer=regularizer)
                # final_vec_wf_3 = tf.matmul(fffinal_vec, wff) + bff

                with tf.name_scope("final"):
                    if graph_hyper_params['mtyp'] == 1:
                        pre_pred_val_deep = tf.split(tf.nn.softmax(final_vec_wf_2), [1, 1], axis=1, name='pred')[0]
                        pre_pred_val = pre_pred_val_deep
                        pred_val = pre_pred_val_deep
                    elif graph_hyper_params['mtyp'] == 2:
                        pre_pred_val_wide = tf.split(tf.nn.softmax(wide_pred), [1, 1], axis=1, name='pred')[0]
                        pre_pred_val = pre_pred_val_wide
                        pred_val = pre_pred_val_wide
                    elif graph_hyper_params['mtyp'] == 3 or graph_hyper_params['mtyp'] == 4:
                        # pre_pred_val_deep = tf.split(tf.nn.softmax(final_vec_wf_2), [1, 1], axis=1)[0]
                        # pre_pred_val_wide = tf.split(tf.nn.softmax(wide_pred), [1, 1], axis=1)[0]
                        # pre_pred_val = tf.div(pre_pred_val_deep + pre_pred_val_wide, 2.0, name='pred')
                        # pred_val = pre_pred_val
                        pre_pred_val = tf.split(tf.nn.softmax(wide_pred+final_vec_wf_2), [1, 1], axis=1, name='pred')[0]
                        # pre_pred_val = pre_pred_val
                        pred_val = pre_pred_val
                    else:
                        print 'wrong model type !'
                    # pre_pred_val_deep = tf.split(tf.nn.softmax(final_vec_wf_2), [1, 1], axis=1)[0]
                    # pre_pred_val_wide = tf.split(tf.nn.softmax(final_vec_wf_2), [1, 1], axis=1)[0]
                    # pre_pred_val = tf.div(pre_pred_val_deep + pre_pred_val_wide, 2.0, name='pred')
                    # pred_val = pre_pred_val
                    # tf.nn.softmax_cross_entropy_with_logits()
                    # pred_val = tf.nn.sigmoid(pre_pred_val, name='pred')
            else:
                fen_zhi = tf.reduce_sum(net_u_final * net_v_final, 1, keep_dims=True)
                norm_u = tf.sqrt(tf.reduce_sum(tf.square(net_u_final), 1, keep_dims=True))
                norm_v = tf.sqrt(tf.reduce_sum(tf.square(net_v_final), 1, keep_dims=True))
                fen_mu = norm_u * norm_v + 1e-6
                pre_pred_val = fen_zhi
                final_vec = fen_mu
                with tf.name_scope("final"):
                    pred_val = tf.nn.relu(fen_zhi / fen_mu, name='pred')

            regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            if graph_hyper_params['mtyp'] == 1:
                gmf_loss = tf.reduce_mean(-true_label * tf.log(pre_pred_val_deep + 1e-6) - (1.0 - true_label) * tf.log(1.0 - pre_pred_val_deep + 1e-6))
            elif graph_hyper_params['mtyp'] == 2:
                gmf_loss = tf.reduce_mean(-true_label * tf.log(pre_pred_val_wide + 1e-6) - (1.0 - true_label) * tf.log(1.0 - pre_pred_val_wide + 1e-6))
            elif graph_hyper_params['mtyp'] == 3 or graph_hyper_params['mtyp'] == 4:
                # gmf_loss_deep = tf.reduce_mean(-true_label * tf.log(pre_pred_val_deep + 1e-6) - (1.0 - true_label) * tf.log(1.0 - pre_pred_val_deep + 1e-6))
                # gmf_loss_wide = tf.reduce_mean(-true_label * tf.log(pre_pred_val_wide + 1e-6) - (1.0 - true_label) * tf.log(1.0 - pre_pred_val_wide + 1e-6))
                # gmf_loss = 0.5 * gmf_loss_deep + 0.5 * gmf_loss_wide
                gmf_loss = tf.reduce_mean(- true_label * tf.log(pre_pred_val + 1e-6)\
                                          - (1.0 - true_label) * tf.log(1.0 - pre_pred_val + 1e-6))
            else:
                print 'wrong model type !'
            return pred_val, gmf_loss + regularization_loss, [pre_pred_val, final_vec, user_vector, ad_vector]

        elif graph_hyper_params['model'] == 'fm':
            print 'FM model !'
            fea_vector = tf.concat([
                lbs_v_ori, age_v_ori, carrier_v_ori, consumptionability_v_ori, education_v_ori, gender_v_ori, house_v_ori, os_v_ori, ct_v_ori,
                marriagestatus_v_ori, appIdAction_v_ori_prod, appIdInstall_v_ori_prod, interest1_v_ori_prod, interest2_v_ori_prod, interest3_v_ori_prod,
                interest4_v_ori_prod, interest5_v_ori_prod, kw1_v_ori_prod, kw2_v_ori_prod, kw3_v_ori_prod, topic1_v_ori_prod, topic2_v_ori_prod, topic3_v_ori_prod,
                advertiserid_v_ori, campaignid_v_ori, creativeid_v_ori, adcategoryid_v_ori, productid_v_ori, producttype_v_ori, creativesize_v_ori
            ], axis=1)

            fea_bias = tf.concat([
                lbs_v_ori_bias, age_v_ori_bias, carrier_v_ori_bias, consumptionability_v_ori_bias, education_v_ori_bias, gender_v_ori_bias,
                house_v_ori_bias, os_v_ori_bias, ct_v_ori_bias,
                marriagestatus_v_ori_bias, appIdAction_v_ori_prod_bias, appIdInstall_v_ori_prod_bias, interest1_v_ori_prod_bias,
                interest2_v_ori_prod_bias, interest3_v_ori_prod_bias,
                interest4_v_ori_prod_bias, interest5_v_ori_prod_bias, kw1_v_ori_prod_bias, kw2_v_ori_prod_bias,
                kw3_v_ori_prod_bias, topic1_v_ori_prod_bias, topic2_v_ori_prod_bias, topic3_v_ori_prod_bias,
                advertiserid_v_ori_bias, campaignid_v_ori_bias, creativeid_v_ori_bias, adcategoryid_v_ori_bias, productid_v_ori_bias, producttype_v_ori_bias, creativesize_v_ori_bias
            ], axis=1)

            summed_features_emb = tf.reduce_sum(fea_vector, 1)
            summed_features_emb_square = tf.square(summed_features_emb)

            squared_features_emb = tf.square(fea_vector)
            squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)

            fm = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
            # if self.batch_norm:
            #     self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
            # self.FM = tf.nn.dropout(self.FM, self.dropout_keep)  # dropout at the FM layer

            bilinear = tf.reduce_sum(fm, 1, keep_dims=True)
            fmb = tf.reduce_sum(fea_bias, 1)

            weight_bias = tf.Variable(tf.constant(0.0), name='fm_out_bias')
            bias_out = weight_bias * tf.ones_like(fmb, dtype=tf.float32)  # None * 1
            out = tf.add_n([bilinear, fmb, bias_out])  # None * 1

            with tf.name_scope("final"):
                out = tf.sigmoid(out, name='pred')

            m_loss = tf.reduce_mean(-true_label * tf.log(out + 1e-10) - (1.0 - true_label) * tf.log(1.0 - out + 1e-10))
            regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return out, m_loss + regularization_loss, []
        elif graph_hyper_params['model'] == 'nfm':
            print 'NFM model !'
            pass
        else:
            print 'Wrong Model !'


    pass



