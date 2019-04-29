# -*- coding: utf-8 -*-
'''
@time: 2018/10/12 19:22
数据的标注非常坑，json里面的imagePath和真实的图片名称对不上，因而采用替换.json为.jpg的方法
@ author: javis
'''

import os
import json
import numpy as np
import glob
import cv2
np.random.seed(41)

#0为背景
classname_to_id = {'1': 1}


class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, text_path):
        self._init_categories()
        label = '1'
        lines = self.read_txtfile(text_path)
        if text_path == './our_train.txt':
            for line in lines:
                lineData = line.strip().split(' 1 ')  # 去除空白和逗号“,”
                imagepath = 'our/'+lineData[0].split(' ')[0]

                image = self._image(imagepath)
                height = image['height']
                width = image['width']
                self.images.append(image)

                for i in range(len(lineData) - 1):
                    points = lineData[i + 1].split(' ')
                    points = [int(i) for i in points]
                    if len(points) < 4:
                        continue
                    elif points[0] + points[2] > width or points[1] + points[3] > height:
                        continue
                    elif points[2] <= 0 or points[3] <= 0:
                        continue
                    else:
                        annotation = self._annotation(label, points)
                        self.annotations.append(annotation)
                        self.ann_id += 1
                self.img_id += 1
        else:
            for line in lines:
                lineData = line.strip().split(' 1 ')  # 去除空白和逗号“,”
                imagepath = lineData[0].split(' ')[0]

                image = self._image(imagepath)
                height = image['height']
                width = image['width']
                self.images.append(image)

                for i in range(len(lineData) - 1):
                    points = lineData[i + 1].split(' ')
                    points = [int(i) for i in points]
                    if len(points) < 4:
                        continue
                    elif points[0] + points[2] > width or points[1] + points[3] > height:
                        continue
                    elif points[2] <= 0 or points[3] <= 0:
                        continue
                    else:
                        annotation = self._annotation(label, points)
                        self.annotations.append(annotation)
                        self.ann_id += 1
                self.img_id += 1

        instance = {}
        instance['info'] = 'huazai_create'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        # for k, v in classname_to_id.items():
        for k in range(1, 2):
            category = {}
            category['id'] = k
            category['name'] = 'defect%d' %k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        img = cv2.imread(path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    # 构建COCO的annotation字段
    def _annotation(self, label, points):
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = classname_to_id[label]
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取json文件，返回一个json对象
    def read_txtfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return f.readlines()

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):

        return [points[0], points[1], points[2], points[3]]


def train_test_split(data, test_size=0.12):
    n_val = int(len(data) * test_size)
    np.random.shuffle(data)
    train_data = data[:-n_val]
    val_data = data[-n_val:]
    return train_data, val_data


if __name__ == '__main__':

    #path = "./All_train.txt"
    #path = './Mall_train.txt'
    #path = './our_train.txt'
    #path = './Part_A_train.txt'
    #path = './Part_B_train.txt'
    path = './all_data.txt'
    #path = './our_part_b_train.txt'
    # 把训练集转化为COCO的json格式
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(path)
    l2c_train.save_coco_json(train_instance, './annotations/instances_train2017_text.json')



