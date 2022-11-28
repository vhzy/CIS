import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
'''
AU_Embedding:

input_channel = 512
mid_channel = 64
input:[4,512,8,8]
output:[4,1]
'''



class Causal_AU_Embedding(nn.Module):
    def __init__(self, input_channel, mid_channel,out_kernel_size):
        super(Causal_AU_Embedding, self).__init__()
        
        self.input_channel = input_channel
        self.mid_channel = mid_channel
        self.out_kernel_size = out_kernel_size

        self.conv1 = nn.Conv2d(self.input_channel, self.mid_channel, kernel_size = 1,padding = 'same')
        self.maxpool = nn.MaxPool2d(kernel_size=7,padding=3,stride=(1,1))
        self.btn = nn.BatchNorm2d(self.mid_channel)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim = 1)
        #这一步也可以池化实现
        self.fc = nn.Linear(self.mid_channel * out_kernel_size * out_kernel_size,1)
        self.sigmoid = nn.Sigmoid()


   

    def forward(self, x):
        #print(x.shape) [4,512,8,8]
        x_conv1 = self.conv1(x)
        x_maxpool = self.maxpool(x_conv1)
        x_btn = self.btn(x_maxpool)
        x_relu1 = self.relu(x_btn)
        x_flatten = self.flatten(x_relu1)
        x_fc = self.fc(x_flatten)
        x_output = self.sigmoid(x_fc)  #这里就是对于第I个AU的预测结果


        return x_output

