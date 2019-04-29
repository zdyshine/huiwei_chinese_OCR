import numpy as np
import pandas as pd
import cv2
import operator
#df = pd.read_csv('./train_lable.csv')
df1 = pd.read_csv('./all_data1.csv')
#df1 = pd.read_csv('./verify_lable0.csv')

filenames = list(set(df1['FileName']))
filenames.sort()
train_txt = open('./train_data.txt', 'w')

for filename in filenames:
    imagepath = './coco/images/' + filename
    image = cv2.imread(imagepath)

    x1 = np.array(df1[df1['FileName'] == filename]['x1'])
    y1 = np.array(df1[df1['FileName'] == filename]['y1'])
    x2 = np.array(df1[df1['FileName'] == filename]['x2'])
    y2 = np.array(df1[df1['FileName'] == filename]['y2'])
    x3 = np.array(df1[df1['FileName'] == filename]['x3'])
    y3 = np.array(df1[df1['FileName'] == filename]['y3'])
    x4 = np.array(df1[df1['FileName'] == filename]['x4'])
    y4 = np.array(df1[df1['FileName'] == filename]['y4'])
    text_list = np.array(df1[df1['FileName'] == filename]['text'])

    for i in range(text_list.shape[0]):
        xmin = np.min([x1[i], x2[i], x3[i], x4[i]])
        ymin = np.min([y1[i], y2[i], y3[i], y4[i]])
        xmax = np.max([x1[i], x2[i], x3[i], x4[i]])
        ymax = np.max([y1[i], y2[i], y3[i], y4[i]])

        image_cut = image[ymin:ymax, xmin:xmax]
        image_cut_rotate = np.rot90(image_cut)

        # image_cut = image[y1[i]:y3[i], x1[i]:x3[i]]
        # cv2.imshow('jj',image_cut_rotate)
        # cv2.waitKey()

        cv2.imwrite('./train_images/'+filename.split('.')[0]+'%d.jpg'%i, image_cut_rotate)
        train_txt.writelines(['./train_images/'+filename.split('.')[0]+'%d.jpg'%i, ' ', text_list[i]])
        train_txt.write('\n')
train_txt.close()
