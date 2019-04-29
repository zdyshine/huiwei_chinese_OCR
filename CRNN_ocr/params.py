import alphabets

train_data = '/train_lmdb'
test_data = '/test_lmdb'

s_train_data = 'to_lmdb/split_lmdb/s_train_lmdb'
s_test_data = './to_lmdb/split_lmdb/s_test_lmdb'

m_train_data = './to_lmdb/split_lmdb/s_train_lmdb'
m_test_data = './to_lmdb/split_lmdb/s_test_lmdb'

l_train_data = './to_lmdb/split_lmdb/l_train_lmdb'
l_test_data = './to_lmdb/split_lmdb/l_test_lmdb'

using_cuda = True
gpu_id = '0'
random_sample = True
keep_ratio = False#按照比例缩放w
adam = True
adadelta = False
sgd = False
saveInterval = 2
valInterval = 3000
n_test_disp = 10
displayInterval = 50 #每训练多少次显示打印
experiment = './expr'
alphabet = alphabets.alphabet
crnn = ''
beta1 =0.5
lr = 0.00001
momentum = 0.9
niter = 96
nh = 512 #256
# imgW = 100
# imgH = 32
imgW = 320
imgH = 32
batchSize = 8
workers = 2

