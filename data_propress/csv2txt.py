import pandas as pd
import numpy as np
import cv2
# data = pd.read_table("./train_labels.csv", sep=",")
# image_names = data['ID'].tolist()
# image_names = list(set(image_names))


def to_widerface(image_names, save_path):
    result = open(save_path, 'w')
    for name in image_names:
        image_path = './images/'+name
        points_list = np.array(data[data['ID'] == name][' Detection'])
        result.write(image_path + ' ' + str(len(points_list)))
        for points in points_list:
            points = points.split(' ')
            result.writelines(
                [' ', str(points[0]), ' ', str(points[1]), ' ', str(int(points[2]) - int(points[0])), ' ', str(int(points[3]) - int(points[1])), ' '])
            result.write("1")
        result.write("\n")
    result.close()


def to_coco(image_names, data, save_path):
    result = open(save_path, 'w')
    for name in image_names:
        image_path = './images/' + name

        x1 = np.array(data[data['FileName'] == name]['x1'])
        y1 = np.array(data[data['FileName'] == name]['y1'])
        x2 = np.array(data[data['FileName'] == name]['x2'])
        y2 = np.array(data[data['FileName'] == name]['y2'])
        x3 = np.array(data[data['FileName'] == name]['x3'])
        y3 = np.array(data[data['FileName'] == name]['y3'])
        x4 = np.array(data[data['FileName'] == name]['x4'])
        y4 = np.array(data[data['FileName'] == name]['y4'])

        points_list = []
        
        for i in range(x1.shape[0]):
            xmin = np.min([x1[i], x2[i], x3[i], x4[i]])
            ymin = np.min([y1[i], y2[i], y3[i], y4[i]])
            xmax = np.max([x1[i], x2[i], x3[i], x4[i]])
            ymax = np.max([y1[i], y2[i], y3[i], y4[i]])
            points_list.append([xmin, ymin, xmax, ymax])


        result.write(image_path + ' ' + str(len(points_list)))
        for points in points_list:
            result.writelines([' ', '1', ' ', str(points[0]), ' ', str(points[1]), ' ', str(int(points[2]) - int(points[0])), ' ', str(int(points[3]) - int(points[1]))])
        result.write("\n")
   
    result.close()

def data_process(image_names, data, save_path):
    # delete the data with the wrong marking format

    problem_name_list = []
    for name in image_names:

        x1 = np.array(data[data['FileName'] == name]['x1'])
        y1 = np.array(data[data['FileName'] == name]['y1'])
        x3 = np.array(data[data['FileName'] == name]['x3'])
        y3 = np.array(data[data['FileName'] == name]['y3'])
        w = x3 - x1
        h = y3 - y1
        if all(np.array(w)>0) and all(np.array(h)>0):
            problem_name_list.append(name)
        else:
            continue



if __name__ == '__main__':

    data = pd.read_csv("./all_data.csv", sep=",")
    filenames = list(set(data['FileName']))
    filenames.sort()
    to_coco(filenames, data, './all_data.txt')
    # to_widerface(image_names, './train1.txt')
