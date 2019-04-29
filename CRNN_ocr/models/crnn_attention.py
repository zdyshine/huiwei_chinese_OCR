import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# CNN attention
'''
https://www.jiqizhixin.com/articles/2019-03-06-14
'''

class Self_Attn(torch.nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class CNN(torch.nn.Module):

    def __init__(self,imgH, nc):
        super(CNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.convolution1 = torch.nn.Conv2d(nc, 64, 3, padding=1)
        self.pooling1 = torch.nn.MaxPool2d(2, stride=2)
        self.convolution2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pooling2 = torch.nn.MaxPool2d(2, stride=2)
        self.convolution3 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.BatchNorm1 = torch.nn.BatchNorm2d(256)
        self.convolution4 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.BatchNorm1 = torch.nn.BatchNorm2d(256)
        self.pooling3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.convolution5 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.BatchNorm1 = torch.nn.BatchNorm2d(512)
        self.convolution6 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.BatchNorm2 = torch.nn.BatchNorm2d(512)
        self.pooling4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.convolution7 = torch.nn.Conv2d(512, 512, 2)
        self.BatchNorm1 = torch.nn.BatchNorm2d(512)

        self.attn1 = Self_Attn(64, 'relu')
        self.attn2 = Self_Attn(128, 'relu')
        self.attn3 = Self_Attn(256, 'relu')
        self.attn4 = Self_Attn(512, 'relu')

    def forward(self, x):
        x = F.relu(self.convolution1(x), inplace=True) # 1-->64
        x = self.pooling1(x)
        x = F.relu(self.convolution2(x), inplace=True) # 64-->128
        x = self.attn2(x)# 128-->128
        x = self.pooling2(x)
        x = F.relu(self.convolution3(x), inplace=True) # 128-->256
        x = F.relu(self.convolution4(x), inplace=True) # 256-->256
        x = self.attn3(x)#256-->256
        x = self.pooling3(x)
        x = F.relu(self.convolution5(x), inplace=True) # 256-->512
        x = F.relu(self.BatchNorm1(x), inplace=True)
        x = F.relu(self.convolution6(x), inplace=True) # 512-->512
        x = self.attn4(x)# 512-->512
        x = F.relu(self.BatchNorm2(x), inplace=True)
        x = self.pooling4(x)
        x = F.relu(self.convolution7(x), inplace=True) # 512-->512
        x = self.attn4(x)  # 512-->512
        return x


class RNN(torch.nn.Module):

    def __init__(self, nclass, nh):
        super(RNN, self).__init__()
        self.Bidirectional_LSTM1 = torch.nn.LSTM(512, nh, bidirectional=True)
        self.embedding1 = torch.nn.Linear(nh * 2, 512)
        self.Bidirectional_LSTM2 = torch.nn.LSTM(512, nh, bidirectional=True)
        self.embedding2 = torch.nn.Linear(nh * 2, nclass)

    def forward(self, x):
        x = self.Bidirectional_LSTM1(x)   # LSTM output: output, (h_n, c_n)
        T, b, h = x[0].size()   # x[0]: (seq_len, batch, num_directions * hidden_size)
        x = self.embedding1(x[0].view(T*b, h))
        x = x.view(T, b, -1)  # [16, b, 512]
        x = self.Bidirectional_LSTM2(x)
        T, b, h = x[0].size()
        x = self.embedding2(x[0].view(T * b, h))
        x = x.view(T, b, -1)
        return x  # [22,b,class_num]

class CRNN(torch.nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = torch.nn.Sequential()
        self.cnn.add_module('vgg_rnn', CNN(imgH, nc))
        self.rnn = torch.nn.Sequential()
        self.rnn.add_module('rnn', RNN(nclass, nh))

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        # print(x.size())
        assert h == 1   # "the height of conv must be 1"
        x = x.squeeze(2)  # remove h dimension, b *512 * width
        x = x.permute(2, 0, 1)  # [w, b, c] = [seq_len, batch, input_size]
        x = self.rnn(x)
        # print(x.size())  # (22, 64, 6736)
        return x

