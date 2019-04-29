'''
把all_data划分成训练集和验证测试集
'''
from sklearn.model_selection import train_test_split

def read_txtfile(path):
    with open(path, "r", encoding='utf-8') as f:
        return f.readlines()
path = './txt/l_train_label.txt'
all_data = read_txtfile(path)
train_data_list, val_data_list = train_test_split(all_data, test_size=0.01, random_state=8888)

with open('./txt/ll_train.txt', 'w') as f0:
    for line in train_data_list:
        f0.write(line)
f0.close()

with open('./txt/ll_test.txt', 'w') as f1:
    for line in val_data_list:
        f1.write(line)
f1.close()
