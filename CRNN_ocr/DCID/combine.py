'''
合并训练集csv和验证集csv
'''

import numpy as np
import pandas as pd
import operator
df = pd.read_csv('./train_lable.csv')
df1 = pd.read_csv('./verify_lable.csv')
# train_file = open('./VOC2007/ImageSets/Main/train.txt', 'w')
# val_file = open('./VOC2007/ImageSets/Main/val.txt', 'w')
# trian_val_file = open('./VOC2007/ImageSets/Main/train_val.txt', 'w')
all_df = pd.merge(df, df1, how='outer')
# train_filenames = list(set(df['FileName']))
# val_filenames = list(set(df1['FileName']))
# train_val_filenames = list(set(all_df['FileName']))
# train_filenames.sort()
# val_filenames.sort()
# train_val_filenames.sort()
all_df.to_csv('./all_data.csv',index=False)

# for file in train_filenames:
#     train_file.write(file.split('.')[0]+'\n')
#
# for file in val_filenames:
#     val_file.write(file.split('.')[0]+'\n')
#
# for file in train_val_filenames:
#     trian_val_file.write(file.split('.')[0]+'\n')