# coding:utf-8

import pandas as pd



def pro_mean(file_list):
    pd_list = [pd.read_csv(f) for f in file_list]
    p1 = pd_list[0]
    for i in range(1, len(pd_list)):
        p1['score'] += pd_list[i]['score']
    p1['score'] /= len(pd_list)
    p1.to_csv('two_mean.csv', index=False)





if __name__ == '__main__':
    file_list = ['/Users/Xuehj/mtyp1-2018-05-17-20-25-18_submission.csv',
                 '/Users/Xuehj/mtyp1-2018-05-17-18-00-38_submission.csv',
                 '/Users/Xuehj/mtyp1-2018-05-17-15-41-30_submission.csv',
                 '/Users/Xuehj/mtyp1-2018-05-17-13-21-49_submission.csv',
                 '/Users/Xuehj/mtyp1-2018-05-17-11-00-34_submission.csv',
                 '/Users/Xuehj/mtyp1-2018-05-17-08-37-16_submission.csv',
                 '/Users/Xuehj/mtyp1-2018-05-17-06-06-53_submission.csv',
                 '/Users/Xuehj/mtyp1-2018-05-17-21-59-09_submission.csv',


                 '/Users/Xuehj/mtyp1-2018-05-17-11-03-53_submission.csv',
                 '/Users/Xuehj/mtyp1-2018-05-17-13-36-58_submission.csv',
                 ]
    pro_mean(file_list)
    pass






















