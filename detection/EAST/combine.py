'''
合并训练集csv和验证集csv
'''

import pandas as pd
df = pd.read_csv('./train_lable.csv')
df1 = pd.read_csv('./verify_lable.csv')

all_df = pd.merge(df, df1, how='outer')

all_df.to_csv('./all_data.csv',index=False)
# from torchvision import transforms as tfs
# import cv2
# import os
# import random
# index = random.randint(1,100)
#
# base_dir = './test_img/'
# out_dir = './tmp/'
#
# def train_tf(x):
#     im_aug = tfs.Compose([
#         tfs.ColorJitter(brightness=1),
#         tfs.ColorJitter(contrast=1),
#         # tfs.ColorJitter(saturation=0.5),
#         tfs.ColorJitter(hue=0.5),
#         # tfs.RandomRotation((-3,3))
#     ])
#     x = im_aug(x)
#     return x
#
# for img in os.listdir(base_dir):
#     image = cv2.imread(base_dir + img)
#     image = train_tf(image)
#     # img = img.replace('.jpg','')
#     cv2.imwrite(out_dir + img,image)

