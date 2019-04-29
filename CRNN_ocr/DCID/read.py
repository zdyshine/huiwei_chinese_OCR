#import os
#import codecs
#imagelist = os.listdir('/media/zdy/新加卷/DYng_Z/2-text_detect/raw_data/test_data/')
#with codecs.open('test_list.txt','a') as f:
#	for image in imagelist:
#		f.write(image + '\n')

import cv2
image = cv2.imread('./crnn_image/img_calligraphy_00001_bg0.jpg')
print(image.shape)

