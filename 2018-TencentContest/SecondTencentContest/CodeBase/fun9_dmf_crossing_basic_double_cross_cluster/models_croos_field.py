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
        emb_size, low_emb_size = 20, 20
        ad_emb_size = 20
    elif 'fm' in graph_hyper_params['model']:
        emb_size, low_emb_size = 150, 150
    else:
        emb_size, low_emb_size = 0, 0
        print 'no this model infer !'
    print emb_size, low_emb_size


    with tf.variable_scope("OutAll"):
        user_field_num, ad_field_num = 23, 7

        user_vector_for_ad_field = {}
        ad_vector_for_user_field = {}

        # user
        for ia in range(ad_field_num):
            # uid_emb = tf.get_variable("uid_emb", shape=(feature_conf_dict['uid'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            LBS_emb = tf.get_variable("LBS_emb"+str(ia), shape=(feature_conf_dict['LBS'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            age_emb = tf.get_variable("age_emb"+str(ia), shape=(feature_conf_dict['age'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            carrier_emb = tf.get_variable("carrier_emb"+str(ia), shape=(feature_conf_dict['carrier'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            consumptionAbility_emb = tf.get_variable("consumptionAbility_emb"+str(ia), shape=(feature_conf_dict['consumptionAbility'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            education_emb = tf.get_variable("education_emb"+str(ia), shape=(feature_conf_dict['education'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            gender_emb = tf.get_variable("gender_emb"+str(ia), shape=(feature_conf_dict['gender'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            house_emb = tf.get_variable("house_emb"+str(ia), shape=(feature_conf_dict['house'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

            os_emb = tf.get_variable("os_emb"+str(ia), shape=(feature_conf_dict['os'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            ct_emb = tf.get_variable("ct_emb"+str(ia), shape=(feature_conf_dict['ct'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            marriageStatus_emb = tf.get_variable("marriageStatus_emb"+str(ia), shape=(feature_conf_dict['marriageStatus'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

            appIdAction_emb = tf.get_variable("appIdAction_emb"+str(ia), shape=(feature_conf_dict['appIdAction'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            appIdInstall_emb = tf.get_variable("appIdInstall_emb"+str(ia), shape=(feature_conf_dict['appIdInstall'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            interest1_emb = tf.get_variable("interest1_emb"+str(ia), shape=(feature_conf_dict['interest1'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            interest2_emb = tf.get_variable("interest2_emb"+str(ia), shape=(feature_conf_dict['interest2'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            interest3_emb = tf.get_variable("interest3_emb"+str(ia), shape=(feature_conf_dict['interest3'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            interest4_emb = tf.get_variable("interest4_emb"+str(ia), shape=(feature_conf_dict['interest4'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,regularizer=regularizer)
            interest5_emb = tf.get_variable("interest5_emb"+str(ia), shape=(feature_conf_dict['interest5'][0], emb_size),initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

            kw1_emb = tf.get_variable("kw1_emb"+str(ia), shape=(feature_conf_dict['kw1'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            kw2_emb = tf.get_variable("kw2_emb"+str(ia), shape=(feature_conf_dict['kw2'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            kw3_emb = tf.get_variable("kw3_emb"+str(ia), shape=(feature_conf_dict['kw3'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

            topic1_emb = tf.get_variable("topic1_emb"+str(ia), shape=(feature_conf_dict['topic1'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            topic2_emb = tf.get_variable("topic2_emb"+str(ia), shape=(feature_conf_dict['topic2'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            topic3_emb = tf.get_variable("topic3_emb"+str(ia), shape=(feature_conf_dict['topic3'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)


            # user
            lbs_v_ori = tf.nn.embedding_lookup(LBS_emb, lbs_p)
            lbs_v_ori_val = tf.expand_dims(tf.cast(tf.cast(lbs_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            lbs_v_ori = tf.matmul(lbs_v_ori_val, lbs_v_ori) # none * 1 * emb_size

            age_v_ori = tf.nn.embedding_lookup(age_emb, age_p)
            age_v_ori_val = tf.expand_dims(tf.cast(tf.cast(age_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            age_v_ori = tf.matmul(age_v_ori_val, age_v_ori)

            carrier_v_ori = tf.nn.embedding_lookup(carrier_emb, carrier_p)
            carrier_v_ori_val = tf.expand_dims(tf.cast(tf.cast(carrier_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            carrier_v_ori = tf.matmul(carrier_v_ori_val, carrier_v_ori)

            consumptionability_v_ori = tf.nn.embedding_lookup(consumptionAbility_emb, consumptionability_p)
            consumptionability_v_ori_val = tf.expand_dims(tf.cast(tf.cast(consumptionability_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            consumptionability_v_ori = tf.matmul(consumptionability_v_ori_val, consumptionability_v_ori)

            education_v_ori = tf.nn.embedding_lookup(education_emb, education_p)
            education_v_ori_val = tf.expand_dims(tf.cast(tf.cast(education_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            education_v_ori = tf.matmul(education_v_ori_val, education_v_ori)

            gender_v_ori = tf.nn.embedding_lookup(gender_emb, gender_p)
            gender_v_ori_val = tf.expand_dims(tf.cast(tf.cast(gender_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            gender_v_ori = tf.matmul(gender_v_ori_val, gender_v_ori)

            house_v_ori = tf.nn.embedding_lookup(house_emb, house_p)
            house_v_ori_val = tf.expand_dims(tf.cast(tf.cast(house_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            house_v_ori = tf.matmul(house_v_ori_val, house_v_ori)

            os_v_ori = tf.nn.embedding_lookup(os_emb, os_p)
            os_v_ori_val = tf.expand_dims(tf.cast(tf.cast(os_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            os_v_ori = tf.matmul(os_v_ori_val, os_v_ori)

            ct_v_ori = tf.nn.embedding_lookup(ct_emb, ct_p)
            ct_v_ori_val = tf.expand_dims(tf.cast(tf.cast(ct_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            ct_v_ori = tf.matmul(ct_v_ori_val, ct_v_ori)

            # marriagestatus_v_ori = tf.nn.embedding_lookup(marriageStatus_emb, marriagestatus_p)
            # marriagestatus_v_ori_val = tf.expand_dims(tf.cast(tf.cast(marriagestatus_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            # marriagestatus_v_ori = tf.matmul(marriagestatus_v_ori_val, marriagestatus_v_ori)
            # marriagestatus_v = tf.reshape(marriagestatus_v_ori, [-1, emb_size])

            appIdAction_v_ori = tf.nn.embedding_lookup(appIdAction_emb, appidaction_index_p)
            appIdAction_v_ori = tf.matmul(appidaction_val_p, appIdAction_v_ori) # none * 1 * emb_size

            appIdInstall_v_ori = tf.nn.embedding_lookup(appIdInstall_emb, appIdInstall_index_p)
            appIdInstall_v_ori = tf.matmul(appIdInstall_val_p, appIdInstall_v_ori)


            marriagestatus_v_ori = tf.nn.embedding_lookup(marriageStatus_emb, marriagestatus_index_p)
            marriagestatus_v_ori_prod = tf.multiply(tf.transpose(marriagestatus_val_p, [0, 2, 1]), marriagestatus_v_ori)
            marriagestatus_v_det = tf.reshape(marriagestatus_v_ori_prod, [-1, feature_conf_dict['marriageStatus'][0], emb_size])


            interest1_v_ori = tf.nn.embedding_lookup(interest1_emb, interest1_index_p)
            interest1_v_ori_prod = tf.multiply(tf.transpose(interest1_val_p, [0, 2, 1]), interest1_v_ori)
            interest1_v_det = tf.reshape(interest1_v_ori_prod, [-1, feature_conf_dict['interest1'][0], emb_size])

            interest2_v_ori = tf.nn.embedding_lookup(interest2_emb, interest2_index_p)
            interest2_v_ori_prod = tf.multiply(tf.transpose(interest2_val_p, [0, 2, 1]), interest2_v_ori)
            interest2_v_det = tf.reshape(interest2_v_ori_prod, [-1, feature_conf_dict['interest2'][0], emb_size])

            interest3_v_ori = tf.nn.embedding_lookup(interest3_emb, interest3_index_p)
            interest3_v_ori_prod = tf.multiply(tf.transpose(interest3_val_p, [0, 2, 1]), interest3_v_ori)
            interest3_v_det = tf.reshape(interest3_v_ori_prod, [-1, feature_conf_dict['interest3'][0], emb_size])

            interest4_v_ori = tf.nn.embedding_lookup(interest4_emb, interest4_index_p)
            interest4_v_ori_prod = tf.multiply(tf.transpose(interest4_val_p, [0, 2, 1]), interest4_v_ori)
            interest4_v_det = tf.reshape(interest4_v_ori_prod, [-1, feature_conf_dict['interest4'][0], emb_size])

            interest5_v_ori = tf.nn.embedding_lookup(interest5_emb, interest5_index_p)
            interest5_v_ori_prod = tf.multiply(tf.transpose(interest5_val_p, [0, 2, 1]), interest5_v_ori)
            interest5_v_det = tf.reshape(interest5_v_ori_prod, [-1, feature_conf_dict['interest5'][0], emb_size])

            kw1_v_ori = tf.nn.embedding_lookup(kw1_emb, kw1_index_p)
            kw1_v = tf.matmul(kw1_val_p, kw1_v_ori)

            kw2_v_ori = tf.nn.embedding_lookup(kw2_emb, kw2_index_p)
            kw2_v = tf.matmul(kw2_val_p, kw2_v_ori)

            kw3_v_ori = tf.nn.embedding_lookup(kw3_emb, kw3_index_p)
            kw3_v = tf.matmul(kw3_val_p, kw3_v_ori)

            topic1_v_ori = tf.nn.embedding_lookup(topic1_emb, topic1_index_p)
            topic1_v = tf.matmul(topic1_val_p, topic1_v_ori)

            topic2_v_ori = tf.nn.embedding_lookup(topic2_emb, topic2_index_p)
            topic2_v = tf.matmul(topic2_val_p, topic2_v_ori)

            topic3_v_ori = tf.nn.embedding_lookup(topic3_emb, topic3_index_p)
            topic3_v = tf.matmul(topic3_val_p, topic3_v_ori)


            user_vector_for_ad_field[str(ia)] = [lbs_v_ori, age_v_ori, carrier_v_ori, consumptionability_v_ori,
                    education_v_ori, gender_v_ori, house_v_ori, os_v_ori, ct_v_ori, appIdAction_v_ori, appIdInstall_v_ori,
                    marriagestatus_v_det, interest1_v_det, interest2_v_det, interest3_v_det, interest4_v_det, interest5_v_det, kw1_v, kw2_v, kw3_v, topic1_v, topic2_v, topic3_v]

            user_vector_for_ad_field[str(ia)+'_size'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    feature_conf_dict['marriageStatus'][0], feature_conf_dict['interest1'][0], feature_conf_dict['interest2'][0],
                    feature_conf_dict['interest3'][0], feature_conf_dict['interest4'][0], feature_conf_dict['interest5'][0],
                    1, 1, 1, 1, 1, 1]


        # ad
        for iu in range(user_field_num):
            # aid_emb = tf.get_variable("aid_emb", shape=(feature_conf_dict['aid'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            advertiserId_emb = tf.get_variable("advertiserId_emb"+str(iu), shape=(feature_conf_dict['advertiserId'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            campaignId_emb = tf.get_variable("campaignId_emb"+str(iu), shape=(feature_conf_dict['campaignId'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            creativeId_emb = tf.get_variable("creativeId_emb"+str(iu), shape=(feature_conf_dict['creativeId'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            adCategoryId_emb = tf.get_variable("adCategoryId_emb"+str(iu), shape=(feature_conf_dict['adCategoryId'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            productId_emb = tf.get_variable("productId_emb"+str(iu), shape=(feature_conf_dict['productId'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            productType_emb = tf.get_variable("productType_emb"+str(iu), shape=(feature_conf_dict['productType'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

            advertiserid_v_ori = tf.nn.embedding_lookup(advertiserId_emb, advertiserid_p)
            advertiserid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(advertiserid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            advertiserid_v_ori = tf.matmul(advertiserid_v_ori_val, advertiserid_v_ori)

            campaignid_v_ori = tf.nn.embedding_lookup(campaignId_emb, campaignid_p)
            campaignid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(campaignid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            campaignid_v_ori = tf.matmul(campaignid_v_ori_val, campaignid_v_ori)

            creativeid_v_ori = tf.nn.embedding_lookup(creativeId_emb, creativeid_p)
            creativeid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(creativeid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            creativeid_v_ori = tf.matmul(creativeid_v_ori_val, creativeid_v_ori)

            adcategoryid_v_ori = tf.nn.embedding_lookup(adCategoryId_emb, adcategoryid_p)
            adcategoryid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(adcategoryid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            adcategoryid_v_ori = tf.matmul(adcategoryid_v_ori_val, adcategoryid_v_ori)

            productid_v_ori = tf.nn.embedding_lookup(productId_emb, productid_p)
            productid_v_ori_val = tf.expand_dims(tf.cast(tf.cast(productid_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            productid_v_ori = tf.matmul(productid_v_ori_val, productid_v_ori)

            producttype_v_ori = tf.nn.embedding_lookup(productType_emb, producttype_p)
            producttype_v_ori_val = tf.expand_dims(tf.cast(tf.cast(producttype_p, dtype=tf.bool), dtype=tf.float32), axis=1)
            producttype_v_ori = tf.matmul(producttype_v_ori_val, producttype_v_ori)

            creativesize_v, creativesize_v_ori, creativesize_v_ori_bias = None, None, None
            if graph_hyper_params['creativeSize_pro'] == 'li_san': # not for dmf
                creativesize_emb = tf.get_variable("creativesize_emb"+str(iu), shape=(feature_conf_dict['creativeSize'], ad_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
                creativesize_v_ori = tf.nn.embedding_lookup(creativesize_emb, creativesize_p)
            else:
                print 'wrong creativeSize_pro'

            ad_vector_for_user_field[str(iu)] = [advertiserid_v_ori, campaignid_v_ori,
                creativeid_v_ori, adcategoryid_v_ori, productid_v_ori, producttype_v_ori, creativesize_v_ori]
            ad_vector_for_user_field[str(iu)+'_size'] = [1, 1, 1, 1, 1, 1, 1]



        # field crossing model
        crossing_result = []
        final_size = 0
        for ia in range(ad_field_num):
            user_vector_this = user_vector_for_ad_field[str(ia)]
            user_vector_size_this = user_vector_for_ad_field[str(ia)+'_size']

            ad_vector_this = []
            ad_vector_size_this = []
            for iu in range(user_field_num):
                ad_vector_this.append(ad_vector_for_user_field[str(iu)][ia])
                ad_vector_size_this.append(ad_vector_for_user_field[str(iu)+'_size'][ia])

            for i in range(user_field_num):
                crossing_result.append(tf.reshape(tf.matmul(user_vector_this[i], tf.transpose(ad_vector_this[i], perm=[0, 2, 1])),
                                                  [-1, user_vector_size_this[i] * ad_vector_size_this[i]]))
                final_size += user_vector_size_this[i] * ad_vector_size_this[i]

        final_result = tf.concat(crossing_result, axis=-1)

        if graph_hyper_params['model'] == 'dmf':
            print 'dmf model !'
            print 'final_size', final_size
            dnn_size = final_size
            wf_1 = tf.get_variable("wf_1", shape=(final_size, dnn_size/2), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            bf_1 = tf.get_variable("bf_1", shape=[dnn_size/2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            final_vec_wf_pre_1 = tf.matmul(final_result, wf_1) + bf_1
            final_vec_wf_1 = tf.nn.relu(final_vec_wf_pre_1)

            wf_2 = tf.get_variable("wf_2", shape=(dnn_size/2, dnn_size / 4), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            bf_2 = tf.get_variable("bf_2", shape=[dnn_size / 4], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            final_vec_wf_2 = tf.matmul(final_vec_wf_1, wf_2) + bf_2
            final_vec_wf_2 = tf.nn.relu(final_vec_wf_2)

            wf_3 = tf.get_variable("wf_3", shape=(dnn_size / 4, 2), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            bf_3 = tf.get_variable("bf_3", shape=[2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            final_vec_wf_3 = tf.matmul(final_vec_wf_2, wf_3) + bf_3

            pre_pred_val_deep = tf.split(tf.nn.softmax(final_vec_wf_3), [1, 1], axis=1, name='pred')[0]
            pre_pred_val = pre_pred_val_deep
            pred_val = pre_pred_val_deep

            gmf_loss = tf.reduce_mean(-true_label * tf.log(pre_pred_val_deep + 1e-6) - (1.0 - true_label) * tf.log(1.0 - pre_pred_val_deep + 1e-6))
            regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return pred_val, gmf_loss + regularization_loss, [pred_val, pre_pred_val, pre_pred_val, pre_pred_val]
    pass



