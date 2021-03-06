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

def inference(uid_p, lbs_p, age_p, carrier_p, consumptionability_p, education_p, gender_p, house_p,
    os_p, ct_p, marriagestatus_index_p, marriagestatus_val_p, appidaction_index_p, appidaction_val_p, appIdInstall_index_p,
    appIdInstall_val_p , interest1_index_p, interest1_val_p, interest2_index_p, interest2_val_p,
    interest3_index_p, interest3_val_p , interest4_index_p, interest4_val_p, interest5_index_p,
    interest5_val_p, kw1_index_p, kw1_val_p, kw2_index_p, kw2_val_p, kw3_index_p, kw3_val_p, topic1_index_p,
    topic1_val_p, topic2_index_p, topic2_val_p, topic3_index_p, topic3_val_p, aid_p, advertiserid_p, campaignid_p,
    creativeid_p,adcategoryid_p, productid_p, producttype_p, creativesize_p, true_label, feature_conf_dict, graph_hyper_params):
    regularizer = tf.contrib.layers.l2_regularizer(graph_hyper_params['l2_reg_alpha'])


    if graph_hyper_params['model'] == 'dmf':
        # emb_size, low_emb_size = 300, 150
        emb_size, low_emb_size = 150, 150
        ad_emb_size = 150
    elif 'fm' in graph_hyper_params['model']:
        emb_size, low_emb_size = 150, 150
    else:
        emb_size, low_emb_size = 0, 0
        print 'no this model infer !'

    print emb_size, low_emb_size

    def gen_basic_relation_mat(name):
        with tf.variable_scope(name):
            LBS_emb = tf.get_variable("LBS_emb", shape=(feature_conf_dict['LBS'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            age_emb = tf.get_variable("age_emb", shape=(feature_conf_dict['age'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            carrier_emb = tf.get_variable("carrier_emb", shape=(feature_conf_dict['carrier'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            consumptionAbility_emb = tf.get_variable("consumptionAbility_emb", shape=(feature_conf_dict['consumptionAbility'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            education_emb = tf.get_variable("education_emb", shape=(feature_conf_dict['education'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            gender_emb = tf.get_variable("gender_emb", shape=(feature_conf_dict['gender'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            house_emb = tf.get_variable("house_emb", shape=(feature_conf_dict['house'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

            os_emb = tf.get_variable("os_emb", shape=(feature_conf_dict['os'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            ct_emb = tf.get_variable("ct_emb", shape=(feature_conf_dict['ct'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            marriageStatus_emb = tf.get_variable("marriageStatus_emb", shape=(feature_conf_dict['marriageStatus'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

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


            # user
            lbs_v_ori = tf.nn.embedding_lookup(LBS_emb, lbs_p)
            lbs_v_ori_val = tf.expand_dims(tf.cast(tf.cast(lbs_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            lbs_v_ori = tf.matmul(lbs_v_ori_val, lbs_v_ori)
            lbs_v = tf.reshape(lbs_v_ori, [-1, emb_size])

            age_v_ori = tf.nn.embedding_lookup(age_emb, age_p)
            age_v_ori_val = tf.expand_dims(tf.cast(tf.cast(age_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            age_v_ori = tf.matmul(age_v_ori_val, age_v_ori)
            age_v = tf.reshape(age_v_ori, [-1, emb_size])

            carrier_v_ori = tf.nn.embedding_lookup(carrier_emb, carrier_p)
            carrier_v_ori_val = tf.expand_dims(tf.cast(tf.cast(carrier_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            carrier_v_ori = tf.matmul(carrier_v_ori_val, carrier_v_ori)
            carrier_v = tf.reshape(carrier_v_ori, [-1, emb_size])

            consumptionability_v_ori = tf.nn.embedding_lookup(consumptionAbility_emb, consumptionability_p)
            consumptionability_v_ori_val = tf.expand_dims(tf.cast(tf.cast(consumptionability_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            consumptionability_v_ori = tf.matmul(consumptionability_v_ori_val, consumptionability_v_ori)
            consumptionability_v = tf.reshape(consumptionability_v_ori, [-1, emb_size])

            education_v_ori = tf.nn.embedding_lookup(education_emb, education_p)
            education_v_ori_val = tf.expand_dims(tf.cast(tf.cast(education_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            education_v_ori = tf.matmul(education_v_ori_val, education_v_ori)
            education_v = tf.reshape(education_v_ori, [-1, emb_size])

            gender_v_ori = tf.nn.embedding_lookup(gender_emb, gender_p)
            gender_v_ori_val = tf.expand_dims(tf.cast(tf.cast(gender_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            gender_v_ori = tf.matmul(gender_v_ori_val, gender_v_ori)
            gender_v = tf.reshape(gender_v_ori, [-1, emb_size])

            house_v_ori = tf.nn.embedding_lookup(house_emb, house_p)
            house_v_ori_val = tf.expand_dims(tf.cast(tf.cast(house_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            house_v_ori = tf.matmul(house_v_ori_val, house_v_ori)
            house_v = tf.reshape(house_v_ori, [-1, emb_size])

            os_v_ori = tf.nn.embedding_lookup(os_emb, os_p)
            os_v_ori_val = tf.expand_dims(tf.cast(tf.cast(os_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            os_v_ori = tf.matmul(os_v_ori_val, os_v_ori)
            os_v = tf.reshape(os_v_ori, [-1, emb_size])

            ct_v_ori = tf.nn.embedding_lookup(ct_emb, ct_p)
            ct_v_ori_val = tf.expand_dims(tf.cast(tf.cast(ct_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            ct_v_ori = tf.matmul(ct_v_ori_val, ct_v_ori)
            ct_v = tf.reshape(ct_v_ori, [-1, emb_size])

            # marriagestatus_v_ori = tf.nn.embedding_lookup(marriageStatus_emb, marriagestatus_p)
            # marriagestatus_v_ori_val = tf.expand_dims(tf.cast(tf.cast(marriagestatus_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            # marriagestatus_v_ori = tf.matmul(marriagestatus_v_ori_val, marriagestatus_v_ori)
            # marriagestatus_v = tf.reshape(marriagestatus_v_ori, [-1, emb_size])

            appIdAction_v_ori = tf.nn.embedding_lookup(appIdAction_emb, appidaction_index_p)
            appIdAction_v = tf.reshape(tf.matmul(appidaction_val_p, appIdAction_v_ori), [-1, emb_size])

            appIdInstall_v_ori = tf.nn.embedding_lookup(appIdInstall_emb, appIdInstall_index_p)
            appIdInstall_v = tf.reshape(tf.matmul(appIdInstall_val_p, appIdInstall_v_ori), [-1, low_emb_size])


            marriagestatus_v_ori = tf.nn.embedding_lookup(marriageStatus_emb, marriagestatus_index_p)
            marriagestatus_v_ori_prod = tf.multiply(tf.transpose(marriagestatus_val_p, [0, 2, 1]), marriagestatus_v_ori)
            marriagestatus_v_det = tf.reshape(marriagestatus_v_ori_prod, [-1, feature_conf_dict['marriageStatus'][0] * emb_size])
            marriagestatus_v = tf.reshape(tf.matmul(marriagestatus_val_p, marriagestatus_v_ori), [-1, emb_size])


            interest1_v_ori = tf.nn.embedding_lookup(interest1_emb, interest1_index_p)
            interest1_v_ori_prod = tf.multiply(tf.transpose(interest1_val_p, [0, 2, 1]), interest1_v_ori)
            interest1_v_det = tf.reshape(interest1_v_ori_prod, [-1, feature_conf_dict['interest1'][0] * emb_size])
            interest1_v = tf.reshape(tf.matmul(interest1_val_p, interest1_v_ori), [-1, emb_size])

            interest2_v_ori = tf.nn.embedding_lookup(interest2_emb, interest2_index_p)
            interest2_v_ori_prod = tf.multiply(tf.transpose(interest2_val_p, [0, 2, 1]), interest2_v_ori)
            interest2_v_det = tf.reshape(interest2_v_ori_prod, [-1, feature_conf_dict['interest2'][0] * emb_size])
            interest2_v = tf.reshape(tf.matmul(interest2_val_p, interest2_v_ori), [-1, emb_size])

            interest3_v_ori = tf.nn.embedding_lookup(interest3_emb, interest3_index_p)
            interest3_v_ori_prod = tf.multiply(tf.transpose(interest3_val_p, [0, 2, 1]), interest3_v_ori)
            interest3_v_det = tf.reshape(interest3_v_ori_prod, [-1, feature_conf_dict['interest3'][0] * emb_size])
            interest3_v = tf.reshape(tf.matmul(interest3_val_p, interest3_v_ori), [-1, emb_size])

            interest4_v_ori = tf.nn.embedding_lookup(interest4_emb, interest4_index_p)
            interest4_v_ori_prod = tf.multiply(tf.transpose(interest4_val_p, [0, 2, 1]), interest4_v_ori)
            interest4_v_det = tf.reshape(interest4_v_ori_prod, [-1, feature_conf_dict['interest4'][0] * emb_size])
            interest4_v = tf.reshape(tf.matmul(interest4_val_p, interest4_v_ori), [-1, emb_size])

            interest5_v_ori = tf.nn.embedding_lookup(interest5_emb, interest5_index_p)
            interest5_v_ori_prod = tf.multiply(tf.transpose(interest5_val_p, [0, 2, 1]), interest5_v_ori)
            interest5_v_det = tf.reshape(interest5_v_ori_prod, [-1, feature_conf_dict['interest5'][0] * emb_size])
            interest5_v = tf.reshape(tf.matmul(interest5_val_p, interest5_v_ori), [-1, emb_size])

            kw1_v_ori= tf.nn.embedding_lookup(kw1_emb, kw1_index_p)
            kw1_v = tf.reshape(tf.matmul(kw1_val_p, kw1_v_ori), [-1, low_emb_size])

            kw2_v_ori = tf.nn.embedding_lookup(kw2_emb, kw2_index_p)
            kw2_v = tf.reshape(tf.matmul(kw2_val_p, kw2_v_ori), [-1, low_emb_size])

            kw3_v_ori = tf.nn.embedding_lookup(kw3_emb, kw3_index_p)
            kw3_v = tf.reshape(tf.matmul(kw3_val_p, kw3_v_ori), [-1, low_emb_size])

            topic1_v_ori = tf.nn.embedding_lookup(topic1_emb, topic1_index_p)
            topic1_v = tf.reshape(tf.matmul(topic1_val_p, topic1_v_ori), [-1, emb_size])

            topic2_v_ori = tf.nn.embedding_lookup(topic2_emb, topic2_index_p)
            topic2_v = tf.reshape(tf.matmul(topic2_val_p, topic2_v_ori), [-1, emb_size])

            topic3_v_ori = tf.nn.embedding_lookup(topic3_emb, topic3_index_p)
            topic3_v = tf.reshape(tf.matmul(topic3_val_p, topic3_v_ori), [-1, emb_size])

            # ad
            advertiserid_v_ori = tf.nn.embedding_lookup(advertiserId_emb, advertiserid_p)
            advertiserid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(advertiserid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            advertiserid_v_ori = tf.matmul(advertiserid_v_ori_val, advertiserid_v_ori)
            advertiserid_v = tf.reshape(advertiserid_v_ori, [-1, ad_emb_size])

            campaignid_v_ori = tf.nn.embedding_lookup(campaignId_emb, campaignid_p)
            campaignid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(campaignid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            campaignid_v_ori = tf.matmul(campaignid_v_ori_val, campaignid_v_ori)
            campaignid_v = tf.reshape(campaignid_v_ori, [-1, ad_emb_size])

            creativeid_v_ori = tf.nn.embedding_lookup(creativeId_emb, creativeid_p)
            creativeid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(creativeid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            creativeid_v_ori = tf.matmul(creativeid_v_ori_val, creativeid_v_ori)
            creativeid_v = tf.reshape(creativeid_v_ori, [-1, ad_emb_size])

            adcategoryid_v_ori = tf.nn.embedding_lookup(adCategoryId_emb, adcategoryid_p)
            adcategoryid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(adcategoryid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            adcategoryid_v_ori = tf.matmul(adcategoryid_v_ori_val, adcategoryid_v_ori)
            adcategoryid_v = tf.reshape(adcategoryid_v_ori, [-1, ad_emb_size])

            productid_v_ori = tf.nn.embedding_lookup(productId_emb, productid_p)
            productid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(productid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            productid_v_ori = tf.matmul(productid_v_ori_val, productid_v_ori)
            productid_v = tf.reshape(productid_v_ori, [-1, ad_emb_size])

            producttype_v_ori = tf.nn.embedding_lookup(productType_emb, producttype_p)
            producttype_v_ori_val = tf.expand_dims(tf.cast(tf.cast(producttype_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            producttype_v_ori = tf.matmul(producttype_v_ori_val, producttype_v_ori)
            producttype_v = tf.reshape(producttype_v_ori, [-1, ad_emb_size])

            creativesize_v, creativesize_v_ori, creativesize_v_ori_bias = None, None, None
            if graph_hyper_params['creativeSize_pro'] == 'min_max':
                creativesize_v = tf.reshape(creativesize_p, [-1, 1])
                ad_vector_size = ad_emb_size * 6 + 1 # for dmf
            elif graph_hyper_params['creativeSize_pro'] == 'li_san': # not for dmf
                creativesize_emb = tf.get_variable("creativesize_emb", shape=(feature_conf_dict['creativeSize'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

                creativesize_v_ori = tf.nn.embedding_lookup(creativesize_emb, creativesize_p)
                creativesize_v = tf.reshape(creativesize_v_ori, [-1, ad_emb_size])
                ad_vector_size = ad_emb_size * 7  # for dmf
            else:
                print 'wrong creativeSize_pro'

            # init vector
            user_vector = tf.concat([lbs_v, age_v, carrier_v, consumptionability_v, education_v, gender_v, house_v, os_v, ct_v,
                                    marriagestatus_v_det, appIdAction_v, appIdInstall_v, interest1_v_det, interest2_v_det, interest3_v_det,
                                    interest4_v_det, interest5_v_det, kw1_v, kw2_v, kw3_v, topic1_v, topic2_v, topic3_v], axis=-1)
            ad_vector = tf.concat([advertiserid_v, campaignid_v, creativeid_v, adcategoryid_v, productid_v, producttype_v, creativesize_v], axis=-1)
            user_vector = tf.reshape(user_vector, [-1, 13 + 4 + (
                        feature_conf_dict['marriageStatus'][0] + feature_conf_dict['interest1'][0] +
                        feature_conf_dict['interest2'][0] + feature_conf_dict['interest3'][0] +
                        feature_conf_dict['interest4'][0] + feature_conf_dict['interest5'][0]), emb_size])
            ad_vector = tf.reshape(ad_vector, [-1, 7, ad_emb_size])

            relation_mat_1 = None
            if graph_hyper_params['model'] == 'dmf':
                print 'dmf model !'
                relation_mat_1 = tf.matmul(user_vector, tf.transpose(ad_vector, perm=[0, 2, 1]))
                uu_s = 13 + 4 + feature_conf_dict['marriageStatus'][0] + feature_conf_dict['interest1'][0] + feature_conf_dict['interest2'][0] + feature_conf_dict['interest3'][0] + feature_conf_dict['interest4'][0] + feature_conf_dict['interest5'][0]
                ad_s = 7
                flat_size = uu_s * ad_s
                relation_mat_1 = tf.reshape(relation_mat_1, [-1, flat_size])
        return relation_mat_1


    bias_size = 10
    print 'bias_size:', bias_size
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
        marriageStatus_emb = tf.get_variable("marriageStatus_emb", shape=(feature_conf_dict['marriageStatus'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

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
        LBS_emb_bias = tf.get_variable("LBS_emb_bias", shape=(feature_conf_dict['LBS'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        age_emb_bias = tf.get_variable("age_emb_bias", shape=(feature_conf_dict['age'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        carrier_emb_bias = tf.get_variable("carrier_emb_bias", shape=(feature_conf_dict['carrier'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        consumptionAbility_emb_bias = tf.get_variable("consumptionAbility_emb_bias", shape=(feature_conf_dict['consumptionAbility'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        education_emb_bias = tf.get_variable("education_emb_bias", shape=(feature_conf_dict['education'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        gender_emb_bias = tf.get_variable("gender_emb_bias", shape=(feature_conf_dict['gender'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        house_emb_bias = tf.get_variable("house_emb_bias", shape=(feature_conf_dict['house'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        os_emb_bias = tf.get_variable("os_emb_bias", shape=(feature_conf_dict['os'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        ct_emb_bias = tf.get_variable("ct_emb_bias", shape=(feature_conf_dict['ct'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        marriageStatus_emb_bias = tf.get_variable("marriageStatus_emb_bias", shape=(feature_conf_dict['marriageStatus'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        appIdAction_emb_bias = tf.get_variable("appIdAction_emb_bias", shape=(feature_conf_dict['appIdAction'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        appIdInstall_emb_bias = tf.get_variable("appIdInstall_emb_bias", shape=(feature_conf_dict['appIdInstall'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest1_emb_bias = tf.get_variable("interest1_emb_bias", shape=(feature_conf_dict['interest1'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest2_emb_bias = tf.get_variable("interest2_emb_bias", shape=(feature_conf_dict['interest2'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest3_emb_bias = tf.get_variable("interest3_emb_bias", shape=(feature_conf_dict['interest3'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest4_emb_bias = tf.get_variable("interest4_emb_bias", shape=(feature_conf_dict['interest4'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,regularizer=regularizer)
        interest5_emb_bias = tf.get_variable("interest5_emb_bias", shape=(feature_conf_dict['interest5'][0], bias_size),initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        kw1_emb_bias = tf.get_variable("kw1_emb_bias", shape=(feature_conf_dict['kw1'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        kw2_emb_bias = tf.get_variable("kw2_emb_bias", shape=(feature_conf_dict['kw2'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        kw3_emb_bias = tf.get_variable("kw3_emb_bias", shape=(feature_conf_dict['kw3'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        topic1_emb_bias = tf.get_variable("topic1_emb_bias", shape=(feature_conf_dict['topic1'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        topic2_emb_bias = tf.get_variable("topic2_emb_bias", shape=(feature_conf_dict['topic2'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        topic3_emb_bias = tf.get_variable("topic3_emb_bias", shape=(feature_conf_dict['topic3'][0], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        # aid_emb = tf.get_variable("aid_emb", shape=(feature_conf_dict['aid'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        advertiserId_emb_bias = tf.get_variable("advertiserId_emb_bias", shape=(feature_conf_dict['advertiserId'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        campaignId_emb_bias = tf.get_variable("campaignId_emb_bias", shape=(feature_conf_dict['campaignId'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        creativeId_emb_bias = tf.get_variable("creativeId_emb_bias", shape=(feature_conf_dict['creativeId'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        adCategoryId_emb_bias = tf.get_variable("adCategoryId_emb_bias", shape=(feature_conf_dict['adCategoryId'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        productId_emb_bias = tf.get_variable("productId_emb_bias", shape=(feature_conf_dict['productId'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        productType_emb_bias = tf.get_variable("productType_emb_bias", shape=(feature_conf_dict['productType'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        # user
        lbs_v_ori = tf.nn.embedding_lookup(LBS_emb, lbs_p)
        lbs_v_ori_val = tf.expand_dims(tf.cast(tf.cast(lbs_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        lbs_v_ori = tf.matmul(lbs_v_ori_val, lbs_v_ori)
        lbs_v = tf.reshape(lbs_v_ori, [-1, emb_size])

        lbs_v_ori_bias = tf.nn.embedding_lookup(LBS_emb_bias, lbs_p)
        lbs_v_ori_bias = tf.matmul(lbs_v_ori_val, lbs_v_ori_bias)
        lbs_v_bias = tf.reshape(lbs_v_ori_bias, [-1, bias_size])

        age_v_ori = tf.nn.embedding_lookup(age_emb, age_p)
        age_v_ori_val = tf.expand_dims(tf.cast(tf.cast(age_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        age_v_ori = tf.matmul(age_v_ori_val, age_v_ori)
        age_v = tf.reshape(age_v_ori, [-1, emb_size])

        age_v_ori_bias = tf.nn.embedding_lookup(age_emb_bias, age_p)
        age_v_ori_bias = tf.matmul(age_v_ori_val, age_v_ori_bias)
        age_v_bias = tf.reshape(age_v_ori_bias, [-1, bias_size])

        carrier_v_ori = tf.nn.embedding_lookup(carrier_emb, carrier_p)
        carrier_v_ori_val = tf.expand_dims(tf.cast(tf.cast(carrier_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        carrier_v_ori = tf.matmul(carrier_v_ori_val, carrier_v_ori)
        carrier_v = tf.reshape(carrier_v_ori, [-1, emb_size])

        carrier_v_ori_bias = tf.nn.embedding_lookup(carrier_emb_bias, carrier_p)
        carrier_v_ori_bias = tf.matmul(carrier_v_ori_val, carrier_v_ori_bias)
        carrier_v_bias = tf.reshape(carrier_v_ori_bias, [-1, bias_size])

        consumptionability_v_ori = tf.nn.embedding_lookup(consumptionAbility_emb, consumptionability_p)
        consumptionability_v_ori_val = tf.expand_dims(tf.cast(tf.cast(consumptionability_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        consumptionability_v_ori = tf.matmul(consumptionability_v_ori_val, consumptionability_v_ori)
        consumptionability_v = tf.reshape(consumptionability_v_ori, [-1, emb_size])

        consumptionability_v_ori_bias = tf.nn.embedding_lookup(consumptionAbility_emb_bias, consumptionability_p)
        consumptionability_v_ori_bias = tf.matmul(consumptionability_v_ori_val, consumptionability_v_ori_bias)
        consumptionability_v_bias = tf.reshape(consumptionability_v_ori_bias, [-1, bias_size])

        education_v_ori = tf.nn.embedding_lookup(education_emb, education_p)
        education_v_ori_val = tf.expand_dims(tf.cast(tf.cast(education_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        education_v_ori = tf.matmul(education_v_ori_val, education_v_ori)
        education_v = tf.reshape(education_v_ori, [-1, emb_size])

        education_v_ori_bias = tf.nn.embedding_lookup(education_emb_bias, education_p)
        education_v_ori_bias = tf.matmul(education_v_ori_val, education_v_ori_bias)
        education_v_bias = tf.reshape(education_v_ori_bias, [-1, bias_size])

        gender_v_ori = tf.nn.embedding_lookup(gender_emb, gender_p)
        gender_v_ori_val = tf.expand_dims(tf.cast(tf.cast(gender_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        gender_v_ori = tf.matmul(gender_v_ori_val, gender_v_ori)
        gender_v = tf.reshape(gender_v_ori, [-1, emb_size])

        gender_v_ori_bias = tf.nn.embedding_lookup(gender_emb_bias, gender_p)
        gender_v_ori_bias = tf.matmul(gender_v_ori_val, gender_v_ori_bias)
        gender_v_bias = tf.reshape(gender_v_ori_bias, [-1, bias_size])

        house_v_ori = tf.nn.embedding_lookup(house_emb, house_p)
        house_v_ori_val = tf.expand_dims(tf.cast(tf.cast(house_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        house_v_ori = tf.matmul(house_v_ori_val, house_v_ori)
        house_v = tf.reshape(house_v_ori, [-1, emb_size])

        house_v_ori_bias = tf.nn.embedding_lookup(house_emb_bias, house_p)
        house_v_ori_bias = tf.matmul(house_v_ori_val, house_v_ori_bias)
        house_v_bias = tf.reshape(house_v_ori_bias, [-1, bias_size])

        os_v_ori = tf.nn.embedding_lookup(os_emb, os_p)
        os_v_ori_val = tf.expand_dims(tf.cast(tf.cast(os_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        os_v_ori = tf.matmul(os_v_ori_val, os_v_ori)
        os_v = tf.reshape(os_v_ori, [-1, emb_size])

        os_v_ori_bias = tf.nn.embedding_lookup(os_emb_bias, os_p)
        os_v_ori_bias = tf.matmul(os_v_ori_val, os_v_ori_bias)
        os_v_bias = tf.reshape(os_v_ori_bias, [-1, bias_size])

        ct_v_ori = tf.nn.embedding_lookup(ct_emb, ct_p)
        ct_v_ori_val = tf.expand_dims(tf.cast(tf.cast(ct_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        ct_v_ori = tf.matmul(ct_v_ori_val, ct_v_ori)
        ct_v = tf.reshape(ct_v_ori, [-1, emb_size])

        ct_v_ori_bias = tf.nn.embedding_lookup(ct_emb_bias, ct_p)
        ct_v_ori_bias = tf.matmul(ct_v_ori_val, ct_v_ori_bias)
        ct_v_bias = tf.reshape(ct_v_ori_bias, [-1, bias_size])

        # marriagestatus_v_ori = tf.nn.embedding_lookup(marriageStatus_emb, marriagestatus_p)
        # marriagestatus_v_ori_val = tf.expand_dims(tf.cast(tf.cast(marriagestatus_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        # marriagestatus_v_ori = tf.matmul(marriagestatus_v_ori_val, marriagestatus_v_ori)
        # marriagestatus_v = tf.reshape(marriagestatus_v_ori, [-1, emb_size])

        appIdAction_v_ori = tf.nn.embedding_lookup(appIdAction_emb, appidaction_index_p)
        appIdAction_v = tf.reshape(tf.matmul(appidaction_val_p, appIdAction_v_ori), [-1, emb_size])

        appIdAction_v_ori_bias = tf.nn.embedding_lookup(appIdAction_emb_bias, appidaction_index_p)
        appIdAction_v_bias = tf.reshape(tf.matmul(appidaction_val_p, appIdAction_v_ori_bias), [-1, bias_size])

        appIdInstall_v_ori = tf.nn.embedding_lookup(appIdInstall_emb, appIdInstall_index_p)
        appIdInstall_v = tf.reshape(tf.matmul(appIdInstall_val_p, appIdInstall_v_ori), [-1, low_emb_size])

        appIdInstall_v_ori_bias = tf.nn.embedding_lookup(appIdInstall_emb_bias, appIdInstall_index_p)
        appIdInstall_v_bias = tf.reshape(tf.matmul(appIdInstall_val_p, appIdInstall_v_ori_bias), [-1, bias_size])

        marriagestatus_v_ori = tf.nn.embedding_lookup(marriageStatus_emb, marriagestatus_index_p)
        marriagestatus_v_ori_prod = tf.multiply(tf.transpose(marriagestatus_val_p, [0, 2, 1]), marriagestatus_v_ori)
        marriagestatus_v_det = tf.reshape(marriagestatus_v_ori_prod, [-1, feature_conf_dict['marriageStatus'][0] * emb_size])
        marriagestatus_v = tf.reshape(tf.matmul(marriagestatus_val_p, marriagestatus_v_ori), [-1, emb_size])

        marriagestatus_v_ori_bias = tf.nn.embedding_lookup(marriageStatus_emb_bias, marriagestatus_index_p)
        marriagestatus_v_ori_prod_bias = tf.multiply(tf.transpose(marriagestatus_val_p, [0, 2, 1]), marriagestatus_v_ori_bias)
        marriagestatus_v_det_bias = tf.reshape(marriagestatus_v_ori_prod_bias, [-1, bias_size*feature_conf_dict['marriageStatus'][0]])
        marriagestatus_v_bias = tf.reshape(tf.matmul(marriagestatus_val_p, marriagestatus_v_ori_bias), [-1, bias_size])

        interest1_v_ori = tf.nn.embedding_lookup(interest1_emb, interest1_index_p)
        interest1_v_ori_prod = tf.multiply(tf.transpose(interest1_val_p, [0, 2, 1]), interest1_v_ori)
        interest1_v_det = tf.reshape(interest1_v_ori_prod, [-1, feature_conf_dict['interest1'][0] * emb_size])
        interest1_v = tf.reshape(tf.matmul(interest1_val_p, interest1_v_ori), [-1, emb_size])

        interest1_v_ori_bias = tf.nn.embedding_lookup(interest1_emb_bias, interest1_index_p)
        interest1_v_ori_prod_bias = tf.multiply(tf.transpose(interest1_val_p, [0, 2, 1]), interest1_v_ori_bias)
        interest1_v_det_bias = tf.reshape(interest1_v_ori_prod_bias, [-1, bias_size*feature_conf_dict['interest1'][0] ])
        interest1_v_bias = tf.reshape(tf.matmul(interest1_val_p, interest1_v_ori_bias), [-1, bias_size])

        interest2_v_ori = tf.nn.embedding_lookup(interest2_emb, interest2_index_p)
        interest2_v_ori_prod = tf.multiply(tf.transpose(interest2_val_p, [0, 2, 1]), interest2_v_ori)
        interest2_v_det = tf.reshape(interest2_v_ori_prod, [-1, feature_conf_dict['interest2'][0] * emb_size])
        interest2_v = tf.reshape(tf.matmul(interest2_val_p, interest2_v_ori), [-1, emb_size])

        interest2_v_ori_bias = tf.nn.embedding_lookup(interest2_emb_bias, interest2_index_p)
        interest2_v_ori_prod_bias = tf.multiply(tf.transpose(interest2_val_p, [0, 2, 1]), interest2_v_ori_bias)
        interest2_v_det_bias = tf.reshape(interest2_v_ori_prod_bias, [-1, bias_size*feature_conf_dict['interest2'][0] ])
        interest2_v_bias = tf.reshape(tf.matmul(interest2_val_p, interest2_v_ori_bias), [-1, bias_size])

        interest3_v_ori = tf.nn.embedding_lookup(interest3_emb, interest3_index_p)
        interest3_v_ori_prod = tf.multiply(tf.transpose(interest3_val_p, [0, 2, 1]), interest3_v_ori)
        interest3_v_det = tf.reshape(interest3_v_ori_prod, [-1, feature_conf_dict['interest3'][0] * emb_size])
        interest3_v = tf.reshape(tf.matmul(interest3_val_p, interest3_v_ori), [-1, emb_size])

        interest3_v_ori_bias = tf.nn.embedding_lookup(interest3_emb_bias, interest3_index_p)
        interest3_v_ori_prod_bias = tf.multiply(tf.transpose(interest3_val_p, [0, 2, 1]), interest3_v_ori_bias)
        interest3_v_det_bias = tf.reshape(interest3_v_ori_prod_bias, [-1, bias_size*feature_conf_dict['interest3'][0]])
        interest3_v_bias = tf.reshape(tf.matmul(interest3_val_p, interest3_v_ori_bias), [-1, bias_size])

        interest4_v_ori = tf.nn.embedding_lookup(interest4_emb, interest4_index_p)
        interest4_v_ori_prod = tf.multiply(tf.transpose(interest4_val_p, [0, 2, 1]), interest4_v_ori)
        interest4_v_det = tf.reshape(interest4_v_ori_prod, [-1, feature_conf_dict['interest4'][0] * emb_size])
        interest4_v = tf.reshape(tf.matmul(interest4_val_p, interest4_v_ori), [-1, emb_size])

        interest4_v_ori_bias = tf.nn.embedding_lookup(interest4_emb_bias, interest4_index_p)
        interest4_v_ori_prod_bias = tf.multiply(tf.transpose(interest4_val_p, [0, 2, 1]), interest4_v_ori_bias)
        interest4_v_det_bias = tf.reshape(interest4_v_ori_prod_bias, [-1, bias_size*feature_conf_dict['interest4'][0]])
        interest4_v_bias = tf.reshape(tf.matmul(interest4_val_p, interest4_v_ori_bias), [-1, bias_size])

        interest5_v_ori = tf.nn.embedding_lookup(interest5_emb, interest5_index_p)
        interest5_v_ori_prod = tf.multiply(tf.transpose(interest5_val_p, [0, 2, 1]), interest5_v_ori)
        interest5_v_det = tf.reshape(interest5_v_ori_prod, [-1, feature_conf_dict['interest5'][0] * emb_size])
        interest5_v = tf.reshape(tf.matmul(interest5_val_p, interest5_v_ori), [-1, emb_size])

        interest5_v_ori_bias = tf.nn.embedding_lookup(interest5_emb_bias, interest5_index_p)
        interest5_v_ori_prod_bias = tf.multiply(tf.transpose(interest5_val_p, [0, 2, 1]), interest5_v_ori_bias)
        interest5_v_det_bias = tf.reshape(interest5_v_ori_prod_bias, [-1, bias_size*feature_conf_dict['interest5'][0]])
        interest5_v_bias = tf.reshape(tf.matmul(interest5_val_p, interest5_v_ori_bias), [-1, bias_size])

        kw1_v_ori= tf.nn.embedding_lookup(kw1_emb, kw1_index_p)
        kw1_v = tf.reshape(tf.matmul(kw1_val_p, kw1_v_ori), [-1, low_emb_size])

        kw1_v_ori_bias = tf.nn.embedding_lookup(kw1_emb_bias, kw1_index_p)
        kw1_v_bias = tf.reshape(tf.matmul(kw1_val_p, kw1_v_ori_bias), [-1, bias_size])

        kw2_v_ori = tf.nn.embedding_lookup(kw2_emb, kw2_index_p)
        kw2_v = tf.reshape(tf.matmul(kw2_val_p, kw2_v_ori), [-1, low_emb_size])

        kw2_v_ori_bias = tf.nn.embedding_lookup(kw2_emb_bias, kw2_index_p)
        kw2_v_bias = tf.reshape(tf.matmul(kw2_val_p, kw2_v_ori_bias), [-1, bias_size])

        kw3_v_ori = tf.nn.embedding_lookup(kw3_emb, kw3_index_p)
        kw3_v = tf.reshape(tf.matmul(kw3_val_p, kw3_v_ori), [-1, low_emb_size])

        kw3_v_ori_bias = tf.nn.embedding_lookup(kw3_emb_bias, kw3_index_p)
        kw3_v_bias = tf.reshape(tf.matmul(kw3_val_p, kw3_v_ori_bias), [-1, bias_size])

        topic1_v_ori = tf.nn.embedding_lookup(topic1_emb, topic1_index_p)
        topic1_v = tf.reshape(tf.matmul(topic1_val_p, topic1_v_ori), [-1, emb_size])

        topic1_v_ori_bias = tf.nn.embedding_lookup(topic1_emb_bias, topic1_index_p)
        topic1_v_bias = tf.reshape(tf.matmul(topic1_val_p, topic1_v_ori_bias), [-1, bias_size])

        topic2_v_ori = tf.nn.embedding_lookup(topic2_emb, topic2_index_p)
        topic2_v = tf.reshape(tf.matmul(topic2_val_p, topic2_v_ori), [-1, emb_size])

        topic2_v_ori_bias = tf.nn.embedding_lookup(topic2_emb_bias, topic2_index_p)
        topic2_v_bias = tf.reshape(tf.matmul(topic2_val_p, topic2_v_ori_bias), [-1, bias_size])

        topic3_v_ori = tf.nn.embedding_lookup(topic3_emb, topic3_index_p)
        topic3_v = tf.reshape(tf.matmul(topic3_val_p, topic3_v_ori), [-1, emb_size])

        topic3_v_ori_bias = tf.nn.embedding_lookup(topic3_emb_bias, topic3_index_p)
        topic3_v_bias = tf.reshape(tf.matmul(topic3_val_p, topic3_v_ori_bias), [-1, bias_size])

        # ad
        advertiserid_v_ori = tf.nn.embedding_lookup(advertiserId_emb, advertiserid_p)
        advertiserid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(advertiserid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        advertiserid_v_ori = tf.matmul(advertiserid_v_ori_val, advertiserid_v_ori)
        advertiserid_v = tf.reshape(advertiserid_v_ori, [-1, ad_emb_size])

        advertiserid_v_ori_bias = tf.nn.embedding_lookup(advertiserId_emb_bias, advertiserid_p)
        advertiserid_v_ori_bias = tf.matmul(advertiserid_v_ori_val, advertiserid_v_ori_bias)
        advertiserid_v_bias = tf.reshape(advertiserid_v_ori_bias, [-1, bias_size])

        campaignid_v_ori = tf.nn.embedding_lookup(campaignId_emb, campaignid_p)
        campaignid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(campaignid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        campaignid_v_ori = tf.matmul(campaignid_v_ori_val, campaignid_v_ori)
        campaignid_v = tf.reshape(campaignid_v_ori, [-1, ad_emb_size])

        campaignid_v_ori_bias = tf.nn.embedding_lookup(campaignId_emb_bias, campaignid_p)
        campaignid_v_ori_bias = tf.matmul(campaignid_v_ori_val, campaignid_v_ori_bias)
        campaignid_v_bias = tf.reshape(campaignid_v_ori_bias, [-1, bias_size])

        creativeid_v_ori = tf.nn.embedding_lookup(creativeId_emb, creativeid_p)
        creativeid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(creativeid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        creativeid_v_ori = tf.matmul(creativeid_v_ori_val, creativeid_v_ori)
        creativeid_v = tf.reshape(creativeid_v_ori, [-1, ad_emb_size])

        creativeid_v_ori_bias = tf.nn.embedding_lookup(creativeId_emb_bias, creativeid_p)
        creativeid_v_ori_bias = tf.matmul(creativeid_v_ori_val, creativeid_v_ori_bias)
        creativeid_v_bias = tf.reshape(creativeid_v_ori_bias, [-1, bias_size])

        adcategoryid_v_ori = tf.nn.embedding_lookup(adCategoryId_emb, adcategoryid_p)
        adcategoryid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(adcategoryid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        adcategoryid_v_ori = tf.matmul(adcategoryid_v_ori_val, adcategoryid_v_ori)
        adcategoryid_v = tf.reshape(adcategoryid_v_ori, [-1, ad_emb_size])

        adcategoryid_v_ori_bias = tf.nn.embedding_lookup(adCategoryId_emb_bias, adcategoryid_p)
        adcategoryid_v_ori_bias = tf.matmul(adcategoryid_v_ori_val, adcategoryid_v_ori_bias)
        adcategoryid_v_bias = tf.reshape(adcategoryid_v_ori_bias, [-1, bias_size])

        productid_v_ori = tf.nn.embedding_lookup(productId_emb, productid_p)
        productid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(productid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        productid_v_ori = tf.matmul(productid_v_ori_val, productid_v_ori)
        productid_v = tf.reshape(productid_v_ori, [-1, ad_emb_size])

        productid_v_ori_bias = tf.nn.embedding_lookup(productId_emb_bias, productid_p)
        productid_v_ori_bias = tf.matmul(productid_v_ori_val, productid_v_ori_bias)
        productid_v_bias = tf.reshape(productid_v_ori_bias, [-1, bias_size])

        producttype_v_ori = tf.nn.embedding_lookup(productType_emb, producttype_p)
        producttype_v_ori_val = tf.expand_dims(tf.cast(tf.cast(producttype_p, dtype=tf.bool), dtype=tf.float32), axis=1)
        producttype_v_ori = tf.matmul(producttype_v_ori_val, producttype_v_ori)
        producttype_v = tf.reshape(producttype_v_ori, [-1, ad_emb_size])

        producttype_v_ori_bias = tf.nn.embedding_lookup(productType_emb_bias, producttype_p)
        producttype_v_ori_bias = tf.matmul(producttype_v_ori_val, producttype_v_ori_bias)
        producttype_v_bias = tf.reshape(producttype_v_ori_bias, [-1, bias_size])

        creativesize_v, creativesize_v_ori, creativesize_v_ori_bias, creativesize_v_bias = None, None, None, None
        if graph_hyper_params['creativeSize_pro'] == 'min_max':
            creativesize_v = tf.reshape(creativesize_p, [-1, 1])
            ad_vector_size = ad_emb_size * 6 + 1 # for dmf
        elif graph_hyper_params['creativeSize_pro'] == 'li_san': # not for dmf
            creativesize_emb = tf.get_variable("creativesize_emb", shape=(feature_conf_dict['creativeSize'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            creativesize_emb_bias = tf.get_variable("creativesize_emb_bias", shape=(feature_conf_dict['creativeSize'], bias_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

            creativesize_v_ori = tf.nn.embedding_lookup(creativesize_emb, creativesize_p)
            creativesize_v = tf.reshape(creativesize_v_ori, [-1, ad_emb_size])

            creativesize_v_ori_bias = tf.nn.embedding_lookup(creativesize_emb_bias, creativesize_p)
            creativesize_v_bias = tf.reshape(creativesize_v_ori_bias, [-1, bias_size])

            ad_vector_size = ad_emb_size * 7  # for dmf
        else:
            print 'wrong creativeSize_pro'

        # init vector
        user_vector = tf.concat([lbs_v, age_v, carrier_v, consumptionability_v, education_v, gender_v, house_v, os_v, ct_v,
                                marriagestatus_v_det, appIdAction_v, appIdInstall_v, interest1_v_det, interest2_v_det, interest3_v_det,
                                interest4_v_det, interest5_v_det, kw1_v, kw2_v, kw3_v, topic1_v, topic2_v, topic3_v], axis=-1)

        user_vector_size = emb_size * 13 + 4 * low_emb_size + (feature_conf_dict['marriageStatus'][0]+ \
                           feature_conf_dict['interest1'][0] + \
                           feature_conf_dict['interest2'][0] + feature_conf_dict['interest3'][0] + feature_conf_dict['interest4'][0] \
                           +feature_conf_dict['interest5'][0]) * emb_size

        ad_vector = tf.concat([advertiserid_v, campaignid_v, creativeid_v, adcategoryid_v, productid_v, producttype_v, creativesize_v], axis=-1)
        print 'user_ad_vector_size:', user_vector_size, ad_vector_size

        if graph_hyper_params['model'] == 'dmf':
            print 'dmf model !'
            user_vector = tf.reshape(user_vector, [-1, 13 + 4 + (feature_conf_dict['marriageStatus'][0] + feature_conf_dict['interest1'][0] + feature_conf_dict['interest2'][0] + feature_conf_dict['interest3'][0] + feature_conf_dict['interest4'][0] + feature_conf_dict['interest5'][0]), emb_size])
            ad_vector = tf.reshape(ad_vector, [-1, 7, ad_emb_size])

            relation_mat = tf.matmul(user_vector, tf.transpose(ad_vector, perm=[0, 2, 1]))

            uu_s = 13 + 4 + feature_conf_dict['marriageStatus'][0] + feature_conf_dict['interest1'][0] + feature_conf_dict['interest2'][0] + feature_conf_dict['interest3'][0] + feature_conf_dict['interest4'][0] + feature_conf_dict['interest5'][0]
            ad_s = 7
            flat_size = uu_s * ad_s
            #
            # print 'uu_s, ad_s: ', uu_s, ad_s
            # print 'flat_size', flat_size
            relation_mat = tf.reshape(relation_mat, [-1, flat_size])


            # fm
            embeddings = tf.concat([user_vector, ad_vector], axis=1)

            # sum_square part
            summed_features_emb = tf.reduce_sum(embeddings, 1)  # None * K
            summed_features_emb_square = tf.square(summed_features_emb)  # None * K

            # square_sum part
            squared_features_emb = tf.square(embeddings)
            squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

            # second order
            y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
            # y_second_order = tf.nn.dropout(y_second_order, dropout_keep_fm[1])  # None * K

            # ---------- Deep component ----------
            y_deep = tf.reshape(embeddings, shape=[-1, (uu_s + ad_s) * emb_size])  # None * (F*K)
            in_size = (uu_s + ad_s) * emb_size
            out_size = 1200
            dwf_1 = tf.get_variable("dwf_1", shape=(in_size, out_size / 2),initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            dbf_1 = tf.get_variable("dbf_1", shape=[out_size / 2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            dfinal_vec_wf_pre_1 = tf.matmul(y_deep, dwf_1) + dbf_1
            dfinal_vec_wf_1 = tf.nn.relu(dfinal_vec_wf_pre_1)

            dwf_2 = tf.get_variable("dwf_2", shape=(out_size / 2, out_size / 4), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            dbf_2 = tf.get_variable("dbf_2", shape=[out_size / 4], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            dfinal_vec_wf_2 = tf.matmul(dfinal_vec_wf_1, dwf_2) + dbf_2

            # dnn
            # user_ad_vector = tf.reshape(tf.concat([user_vector, ad_vector], axis=1), [-1, (uu_s + ad_s) * emb_size])
            # dwf_nn_1 = tf.get_variable("dwf_nn_1", shape=((uu_s + ad_s) * emb_size, 1000),initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,regularizer=regularizer)
            # dbf_nn_1 = tf.get_variable("dbf_nn_1", shape=[1000], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32, regularizer=regularizer)
            # dfinal_vec_wf_pre_nn_1 = tf.matmul(user_ad_vector, dwf_nn_1) + dbf_nn_1
            # dfinal_vec_wf_nn_1 = tf.nn.relu(dfinal_vec_wf_pre_nn_1)
            #
            # dwf_nn_2 = tf.get_variable("dwf_nn_2", shape=(1000, 500), initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32, regularizer=regularizer)
            # dbf_nn_2 = tf.get_variable("dbf_nn_2", shape=[500], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32, regularizer=regularizer)
            # dfinal_vec_wf_nn_2 = tf.nn.relu(tf.matmul(dfinal_vec_wf_nn_1, dwf_nn_2) + dbf_nn_2)

            # bias
            user_vector_bias = tf.concat([lbs_v_bias, age_v_bias, carrier_v_bias, consumptionability_v_bias, education_v_bias, gender_v_bias,
                 house_v_bias, os_v_bias, ct_v_bias, marriagestatus_v_det_bias, appIdAction_v_bias, appIdInstall_v_bias, interest1_v_det_bias,
                 interest2_v_det_bias, interest3_v_det_bias, interest4_v_det_bias, interest5_v_det_bias, kw1_v_bias, kw2_v_bias, kw3_v_bias, topic1_v_bias, topic2_v_bias, topic3_v_bias], axis=-1)
            ad_vector_bias = tf.concat([advertiserid_v_bias, campaignid_v_bias, creativeid_v_bias, adcategoryid_v_bias, productid_v_bias, producttype_v_bias, creativesize_v_bias], axis=-1)
            user_vector_bias = tf.reshape(user_vector_bias, [-1, uu_s * bias_size])
            ad_vector_bias = tf.reshape(ad_vector_bias, [-1, ad_s * bias_size])

            # relation_mat = tf.concat([gen_basic_relation_mat("SecondOutAll"),\
            #                           y_second_order, dfinal_vec_wf_2, user_vector_bias, ad_vector_bias], axis=-1)

            # flat_size = emb_size + 300 + uu_s * bias_size + ad_s * bias_size + uu_s * ad_s
            # flat_size = uu_s * ad_s
            final_size = flat_size
            print 'final_flat_size:', flat_size
            wf_1 = tf.get_variable("wf_1", shape=(flat_size, final_size/2), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            bf_1 = tf.get_variable("bf_1", shape=[final_size/2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            final_vec_wf_pre_1 = tf.matmul(relation_mat, wf_1) + bf_1
            final_vec_wf_1 = tf.nn.relu(final_vec_wf_pre_1)

            wf_2 = tf.get_variable("wf_2", shape=(final_size/2, final_size / 4), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            bf_2 = tf.get_variable("bf_2", shape=[final_size / 4], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            final_vec_wf_2 = tf.matmul(final_vec_wf_1, wf_2) + bf_2
            # final_vec_wf_2 = tf.nn.relu(final_vec_wf_2)



            final_vec = tf.concat([final_vec_wf_2, user_vector_bias, ad_vector_bias, \
                                   y_second_order, dfinal_vec_wf_2], axis=-1)
            final_vec_size = flat_size/4 + uu_s * bias_size + ad_s * bias_size + emb_size + 300
            wf_3 = tf.get_variable("wf_3", shape=(final_vec_size, 2), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            bf_3 = tf.get_variable("bf_3", shape=[2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            final_vec_wf_3 = tf.matmul(final_vec, wf_3) + bf_3

            pre_pred_val_deep = tf.split(tf.nn.softmax(final_vec_wf_3), [1, 1], axis=1, name='pred')[0]
            pre_pred_val = pre_pred_val_deep
            pred_val = pre_pred_val_deep

            gmf_loss = tf.reduce_mean(-true_label * tf.log(pre_pred_val_deep + 1e-6) - (1.0 - true_label) * tf.log(1.0 - pre_pred_val_deep + 1e-6))
            regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return pred_val, gmf_loss + regularization_loss, [pred_val, pre_pred_val, user_vector, ad_vector]
    pass



