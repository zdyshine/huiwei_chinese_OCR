#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as tfs
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import os
import random
def train_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize(random.randint(40,80)),
        tfs.ColorJitter(brightness=1),
        tfs.ColorJitter(contrast=1),
        # tfs.ColorJitter(saturation=0.5),
        tfs.ColorJitter(hue=0.5),
        tfs.RandomRotation((-3,3))
    ])
    x = im_aug(x)
    return x

for img in os.listdir('./img'):
    img = Image.open('./img/'+img)
    name = random.randint(1,300)
    name = str(name)
    img = train_tf(img)
    # r,g,b = img.split()
    img.save('./img0/'+name+'.jpg')

# img = Image.open('img/img_calligraphy_00052_bg0.jpg')
#
# n_img = tfs.Resize(30)(img)
# n_img.save('img0/result_19.jpg')
# img = tfs.LinearTransformation()(img)
# img.save('img0/result_17.jpg')
# format_string = train_tf(img)
# format_string.save('img0/result_6.jpg')
# format_string = tfs.ColorJitter(brightness = 1, contrast = 1, hue = 0.5)(img)
# format_string.save('img0/result_10.jpg')
# format_string1 = tfs.ColorJitter(brightness=1)(img)
# format_string2 = tfs.ColorJitter(contrast=1)(img)
# format_string3 = tfs.ColorJitter(saturation=0.5)(img)
# format_string4 = tfs.ColorJitter(hue=0.5)(img)
# # print(format_string.shape)
# format_string1.save('img0/result_1.jpg')
# format_string2.save('img0/result_2.jpg')
# format_string3.save('img0/result_3.jpg')
# format_string4.save('img0/result_4.jpg')