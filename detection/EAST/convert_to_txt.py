# -*- coding: utf-8 -*-
import csv
import os

res = {}
base_dir = '/media/zdy/新加卷/DYng_Z/2-text_detect/data'

def get_annotations(path):
  with open(path, "r") as f:
    reader = csv.reader(f)
    for item in reader:
      if not item[0].endswith('jpg'):
        continue
      if item[0] not in res:
        res[item[0]] = []
      res[item[0]].append(item[1:])
  return res

def write_txt(d, path):
  for name, objects in d.items():
    name = name.split('.')[0] + '.txt'
    with open(os.path.join(path, name), 'w') as f:
      for ob in objects:
        f.write(','.join(ob) + '\n')


if __name__ == '__main__':
  path =base_dir +  '/all_data.csv'
  save_path =base_dir + '/verifyImage/'
  d = get_annotations(path)
  write_txt(d, save_path)
