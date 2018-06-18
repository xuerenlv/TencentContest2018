# coding:utf-8
import tensorflow as tf



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
    os_p, ct_p, marriagestatus_p, appidaction_index_p, appidaction_val_p, appIdInstall_index_p,
    appIdInstall_val_p , interest1_index_p, interest1_val_p, interest2_index_p, interest2_val_p,
    interest3_index_p, interest3_val_p , interest4_index_p, interest4_val_p, interest5_index_p,
    interest5_val_p, kw1_index_p, kw1_val_p, kw2_index_p, kw2_val_p, kw3_index_p, kw3_val_p, topic1_index_p,
    topic1_val_p, topic2_index_p, topic2_val_p, topic3_index_p, topic3_val_p, aid_p, advertiserid_p, campaignid_p,
    creativeid_p,adcategoryid_p, productid_p, producttype_p, creativesize_p, true_label, feature_conf_dict, graph_hyper_params):
    regularizer = tf.contrib.layers.l2_regularizer(graph_hyper_params['l2_reg_alpha'])


    if graph_hyper_params['model'] == 'dmf':
        emb_size = 200
        low_emb_size = 100
    elif 'fm' in graph_hyper_params['model']:
        emb_size, low_emb_size = 100, 100
    else:
        emb_size, low_emb_size = 0, 0
        print 'no this model infer !'

    print emb_size, low_emb_size
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

        appIdAction_emb = tf.get_variable("appIdAction_emb", shape=(feature_conf_dict['appIdAction'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        appIdInstall_emb = tf.get_variable("appIdInstall_emb", shape=(feature_conf_dict['appIdInstall'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest1_emb = tf.get_variable("interest1_emb", shape=(feature_conf_dict['interest1'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest2_emb = tf.get_variable("interest2_emb", shape=(feature_conf_dict['interest2'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest3_emb = tf.get_variable("interest3_emb", shape=(feature_conf_dict['interest3'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        interest4_emb = tf.get_variable("interest4_emb", shape=(feature_conf_dict['interest4'][0], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,regularizer=regularizer)
        interest5_emb = tf.get_variable("interest5_emb", shape=(feature_conf_dict['interest5'][0], emb_size),initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        kw1_emb = tf.get_variable("kw1_emb", shape=(feature_conf_dict['kw1'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        kw2_emb = tf.get_variable("kw2_emb", shape=(feature_conf_dict['kw2'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        kw3_emb = tf.get_variable("kw3_emb", shape=(feature_conf_dict['kw3'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        topic1_emb = tf.get_variable("topic1_emb", shape=(feature_conf_dict['topic1'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        topic2_emb = tf.get_variable("topic2_emb", shape=(feature_conf_dict['topic2'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        topic3_emb = tf.get_variable("topic3_emb", shape=(feature_conf_dict['topic3'][0], low_emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

        # aid_emb = tf.get_variable("aid_emb", shape=(feature_conf_dict['aid'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        advertiserId_emb = tf.get_variable("advertiserId_emb", shape=(feature_conf_dict['advertiserId'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        campaignId_emb = tf.get_variable("campaignId_emb", shape=(feature_conf_dict['campaignId'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        creativeId_emb = tf.get_variable("creativeId_emb", shape=(feature_conf_dict['creativeId'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        adCategoryId_emb = tf.get_variable("adCategoryId_emb", shape=(feature_conf_dict['adCategoryId'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        productId_emb = tf.get_variable("productId_emb", shape=(feature_conf_dict['productId'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
        productType_emb = tf.get_variable("productType_emb", shape=(feature_conf_dict['productType'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

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
        appIdAction_v = tf.reshape(tf.matmul(appidaction_val_p, appIdAction_v_ori), [-1, low_emb_size])

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

        interest2_v_ori = tf.nn.embedding_lookup(interest2_emb, interest2_index_p)
        interest2_v_ori_bias = tf.nn.embedding_lookup(interest2_emb_bias, interest2_index_p)
        interest2_v_ori_prod = tf.multiply(tf.transpose(interest2_val_p, [0, 2, 1]), interest2_v_ori)
        interest2_v_ori_prod_bias = tf.multiply(tf.transpose(interest2_val_p, [0, 2, 1]), interest2_v_ori_bias)
        interest2_v = tf.reshape(tf.matmul(interest2_val_p, interest2_v_ori), [-1, emb_size])

        interest3_v_ori = tf.nn.embedding_lookup(interest3_emb, interest3_index_p)
        interest3_v_ori_bias = tf.nn.embedding_lookup(interest3_emb_bias, interest3_index_p)
        interest3_v_ori_prod = tf.multiply(tf.transpose(interest3_val_p, [0, 2, 1]), interest3_v_ori)
        interest3_v_ori_prod_bias = tf.multiply(tf.transpose(interest3_val_p, [0, 2, 1]), interest3_v_ori_bias)
        interest3_v = tf.reshape(tf.matmul(interest3_val_p, interest3_v_ori), [-1, emb_size])

        interest4_v_ori = tf.nn.embedding_lookup(interest4_emb, interest4_index_p)
        interest4_v_ori_bias = tf.nn.embedding_lookup(interest4_emb_bias, interest4_index_p)
        interest4_v_ori_prod = tf.multiply(tf.transpose(interest4_val_p, [0, 2, 1]), interest4_v_ori)
        interest4_v_ori_prod_bias = tf.multiply(tf.transpose(interest4_val_p, [0, 2, 1]), interest4_v_ori_bias)
        interest4_v = tf.reshape(tf.matmul(interest4_val_p, interest4_v_ori), [-1, emb_size])

        interest5_v_ori = tf.nn.embedding_lookup(interest5_emb, interest5_index_p)
        interest5_v_ori_bias = tf.nn.embedding_lookup(interest5_emb_bias, interest5_index_p)
        interest5_v_ori_prod = tf.multiply(tf.transpose(interest5_val_p, [0, 2, 1]), interest5_v_ori)
        interest5_v_ori_prod_bias = tf.multiply(tf.transpose(interest5_val_p, [0, 2, 1]), interest5_v_ori_bias)
        interest5_v = tf.reshape(tf.matmul(interest5_val_p, interest5_v_ori), [-1, emb_size])

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
        topic1_v = tf.reshape(tf.matmul(topic1_val_p, topic1_v_ori), [-1, low_emb_size])

        topic2_v_ori = tf.nn.embedding_lookup(topic2_emb, topic2_index_p)
        topic2_v_ori_bias = tf.nn.embedding_lookup(topic2_emb_bias, topic2_index_p)
        topic2_v_ori_prod = tf.multiply(tf.transpose(topic2_val_p, [0, 2, 1]), topic2_v_ori)
        topic2_v_ori_prod_bias = tf.multiply(tf.transpose(topic2_val_p, [0, 2, 1]), topic2_v_ori_bias)
        topic2_v = tf.reshape(tf.matmul(topic2_val_p, topic2_v_ori), [-1, low_emb_size])

        topic3_v_ori = tf.nn.embedding_lookup(topic3_emb, topic3_index_p)
        topic3_v_ori_bias = tf.nn.embedding_lookup(topic3_emb_bias, topic3_index_p)
        topic3_v_ori_prod = tf.multiply(tf.transpose(topic3_val_p, [0, 2, 1]), topic3_v_ori)
        topic3_v_ori_prod_bias = tf.multiply(tf.transpose(topic3_val_p, [0, 2, 1]), topic3_v_ori_bias)
        topic3_v = tf.reshape(tf.matmul(topic3_val_p, topic3_v_ori), [-1, low_emb_size])

        # ad
        advertiserid_v_ori = tf.nn.embedding_lookup(advertiserId_emb, advertiserid_p)
        advertiserid_v_ori_bias = tf.nn.embedding_lookup(advertiserId_emb_bias, advertiserid_p)
        advertiserid_v = tf.reshape(advertiserid_v_ori, [-1, emb_size])

        campaignid_v_ori = tf.nn.embedding_lookup(campaignId_emb, campaignid_p)
        campaignid_v_ori_bias = tf.nn.embedding_lookup(campaignId_emb_bias, campaignid_p)
        campaignid_v = tf.reshape(campaignid_v_ori, [-1, emb_size])

        creativeid_v_ori = tf.nn.embedding_lookup(creativeId_emb, creativeid_p)
        creativeid_v_ori_bias = tf.nn.embedding_lookup(creativeId_emb_bias, creativeid_p)
        creativeid_v = tf.reshape(creativeid_v_ori, [-1, emb_size])

        adcategoryid_v_ori = tf.nn.embedding_lookup(adCategoryId_emb, adcategoryid_p)
        adcategoryid_v_ori_bias = tf.nn.embedding_lookup(adCategoryId_emb_bias, adcategoryid_p)
        adcategoryid_v = tf.reshape(adcategoryid_v_ori, [-1, emb_size])

        productid_v_ori = tf.nn.embedding_lookup(productId_emb, productid_p)
        productid_v_ori_bias = tf.nn.embedding_lookup(productId_emb_bias, productid_p)
        productid_v = tf.reshape(productid_v_ori, [-1, emb_size])

        producttype_v_ori = tf.nn.embedding_lookup(productType_emb, producttype_p)
        producttype_v_ori_bias = tf.nn.embedding_lookup(productType_emb_bias, producttype_p)
        producttype_v = tf.reshape(producttype_v_ori, [-1, emb_size])

        creativesize_v, creativesize_v_ori, creativesize_v_ori_bias = None, None, None
        if graph_hyper_params['creativeSize_pro'] == 'min_max':
            creativesize_v = tf.reshape(creativesize_p, [-1, 1])
            ad_vector_size = emb_size * 6 + 1 # for dmf
        elif graph_hyper_params['creativeSize_pro'] == 'li_san': # not for dmf
            creativesize_emb = tf.get_variable("creativesize_emb", shape=(feature_conf_dict['creativeSize'], emb_size), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            creativesize_emb_bias = tf.get_variable("creativesize_emb_bias", shape=(feature_conf_dict['creativeSize'], 1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

            creativesize_v_ori = tf.nn.embedding_lookup(creativesize_emb, creativesize_p)
            creativesize_v_ori_bias = tf.nn.embedding_lookup(creativesize_emb_bias, creativesize_p)

            creativesize_v = tf.reshape(creativesize_v_ori, [-1, emb_size])
            ad_vector_size = emb_size * 7  # for dmf
        else:
            print 'wrong creativeSize_pro'

        # init vector
        user_vector = tf.concat([lbs_v, age_v, carrier_v, consumptionability_v, education_v, gender_v, house_v, os_v, ct_v,
                                marriagestatus_v, appIdAction_v, appIdInstall_v, interest1_v, interest2_v, interest3_v,
                                interest4_v, interest5_v, kw1_v, kw2_v, kw3_v, topic1_v, topic2_v, topic3_v], axis=-1)
        user_vector_size = emb_size * 15 + 8 * low_emb_size
        ad_vector = tf.concat([advertiserid_v, campaignid_v, creativeid_v, adcategoryid_v, productid_v, producttype_v, creativesize_v], axis=-1)

        if graph_hyper_params['model'] == 'dmf':
            print 'dmf model !'
            # network
            u_b1 = tf.get_variable("u_b1", shape=[user_vector_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)

            net_u_1_pre = user_vector + u_b1
            net_u_1 = tf.nn.relu(net_u_1_pre)

            u_w2 = tf.get_variable("u_w2", shape=(user_vector_size, 300), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            u_b2 = tf.get_variable("u_b2", shape=[300], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            net_u_2_pre = tf.matmul(net_u_1, u_w2) + u_b2
            net_u_2 = tf.nn.relu(net_u_2_pre)

            u_w3 = tf.get_variable("u_w3", shape=(300, 300), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            u_b3 = tf.get_variable("u_b3", shape=[300], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            # net_u_final = tf.matmul(net_u_2, u_w3) + u_b3 + net_u_2_pre



            v_b1 = tf.get_variable("v_b1", shape=[ad_vector_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            net_v_1_pre = ad_vector + v_b1
            net_v_1 = tf.nn.relu(net_v_1_pre)

            v_w2 = tf.get_variable("v_w2", shape=(ad_vector_size, 300), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            v_b2 = tf.get_variable("v_b2", shape=[300], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            net_v_2_pre = tf.matmul(net_v_1, v_w2) + v_b2
            net_v_2 = tf.nn.relu(net_v_2_pre)

            v_w3 = tf.get_variable("v_w3", shape=(300, 300), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            v_b3 = tf.get_variable("v_b3", shape=[300], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, regularizer=regularizer)
            # net_v_final = tf.matmul(net_v_2, v_w3) + v_b3 + net_v_2_pre


            print 'dmf 2layer'
            net_u_final = net_u_2_pre
            net_v_final = net_v_2_pre



            fen_zhi = tf.reduce_sum(net_u_final * net_v_final, 1, keep_dims=True)
            norm_u = tf.sqrt(tf.reduce_sum(tf.square(net_u_final), 1, keep_dims=True))
            norm_v = tf.sqrt(tf.reduce_sum(tf.square(net_v_final), 1, keep_dims=True))
            fen_mu = norm_u * norm_v

            with tf.name_scope("final"):
                pred_val = tf.nn.relu(fen_zhi / fen_mu, name='pred')

            regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            gmf_loss = tf.reduce_mean(-true_label * tf.log(pred_val + 1e-10) - (1.0 - true_label) * tf.log(1.0 - pred_val + 1e-10))

            return pred_val, gmf_loss + regularization_loss, []
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
            # m_loss = tf.nn.l2_loss(true_label - out)
            # if self.loss_type == 'square_loss':
            #     if self.lamda_bilinear > 0:
            #         self.loss = tf.nn.l2_loss(tf.sub(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
            #             self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
            #     else:
            #         self.loss = tf.nn.l2_loss(tf.sub(self.train_labels, self.out))
            # elif self.loss_type == 'log_loss':
            #     self.out = tf.sigmoid(self.out)
            #     if self.lambda_bilinear > 0:
            #         self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07,
            #                                                scope=None) + tf.contrib.layers.l2_regularizer(
            #             self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
            #     else:
            #         self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07,
            #                                                scope=None)

            regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return out, m_loss + regularization_loss, []
        elif graph_hyper_params['model'] == 'nfm':
            print 'NFM model !'
            pass
        else:
            print 'Wrong Model !'


    pass



