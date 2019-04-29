'''
@ author zdy
@time 2019.3.19
分析合并后的数据，去掉不合适的
'''

import numpy as np
import pandas as pd
import cv2
import operator

base_dir = '/media/zdy/新加卷/DYng_Z/2-text_detect/data'
image_path = '/home/zdy/DYng/detectron/data/coco'
#df = pd.read_csv('./train_lable.csv')
#df1 = pd.read_csv(base_dir + '/train_lable0.csv')
# df1 = pd.read_csv(base_dir + '/all_data.csv')
df1 = pd.read_csv(base_dir + '/verify_lable.csv') # verify

H = []
W = []
F = []

filenames = list(set(df1['FileName']))
filenames.sort()
train_txt = open('./train_data.txt', 'w')

for filename in filenames:
    imagepath = image_path +'/images/' + filename
    # print(imagepath)
    image = cv2.imread(imagepath)
    # print(image.shape)

    x1 = np.array(df1[df1['FileName'] == filename]['x1'])
    y1 = np.array(df1[df1['FileName'] == filename]['y1'])
    x2 = np.array(df1[df1['FileName'] == filename]['x2'])
    y2 = np.array(df1[df1['FileName'] == filename]['y2'])
    x3 = np.array(df1[df1['FileName'] == filename]['x3'])
    y3 = np.array(df1[df1['FileName'] == filename]['y3'])
    x4 = np.array(df1[df1['FileName'] == filename]['x4'])
    y4 = np.array(df1[df1['FileName'] == filename]['y4'])
    text_list = np.array(df1[df1['FileName'] == filename]['text'])

    for i in range(x1.shape[0]):
        xmin = np.min([x1[i], x2[i], x3[i], x4[i]])
        ymin = np.min([y1[i], y2[i], y3[i], y4[i]])
        xmax = np.max([x1[i], x2[i], x3[i], x4[i]])
        ymax = np.max([y1[i], y2[i], y3[i], y4[i]])
        image_cut = image[ymin:ymax, xmin:xmax]
        image_cut_rotate = np.rot90(image_cut)

        image_cut = image[y1[i]:y3[i], x1[i]:x3[i]]
        # print(image_cut.shape)
        h = image_cut.shape[0]
        w = image_cut.shape[1]

        #cv2.imshow('jj',image_cut_rotate)
        #cv2.waitKey()
        im_name = filename.split('.')[0]+ str(i) + '.jpg'
        F.append(im_name)
        H.append(h)
        W.append(w)
        cv2.imwrite('crnn_image/'+filename.split('.')[0]+'%d.jpg'%i, image_cut_rotate)
        train_txt.writelines(['crnn_image/'+filename.split('.')[0]+'%d.jpg'%i, ' ', text_list[i]])
        train_txt.write('\n')
train_txt.close()
train_size = pd.DataFrame(columns=['name'])
train_size['name'] = F
train_size['Hsize'] = H
train_size['Wsize'] = W
train_size.to_csv('./data_size.csv', header=True, index=None, encoding='utf_8_sig')