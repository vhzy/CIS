import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
'''
attentionbranch:
resnet最后一层输出是512d,attention_branch产生N(AU)个feature map考虑到batch_size,dim:[bn,N,8,8]，比如说[4,512,8,8],
另一边的输出就是[4,12]直接和loss计算相似度
input_channel = 512
output_channel = 64
'''



class AttentionBranchLayer(nn.Module):
    def __init__(self, input_channel, output_channel,au_num):
        super(AttentionBranchLayer, self).__init__()
        
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.au_num = au_num

        self.conv1 = nn.Conv2d(self.input_channel, self.output_channel, kernel_size = 3,padding = 'same')
        self.conv2 = nn.Conv2d(self.output_channel, self.au_num, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv_att = nn.Conv2d(self.au_num,self.au_num,kernel_size=1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv_output = nn.Conv2d(self.au_num,self.au_num,kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid2 = nn.Sigmoid()

   

    def forward(self, x):
        #print(x.shape) [4,512,8,8]
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv2_relu = self.relu1(x_conv2)


        #attention_branch
        x_att = self.conv_att(x_conv2_relu)
        #x_att_relu = self.sigmoid1(x_att)
        attention = self.sigmoid1(x_att)

        #predict branch
        x_conv_output = self.conv_output(x_conv2_relu)
        x_gap = self.avgpool(x_conv_output)
        x_predict = x_gap.view(x.shape[0], -1)
        x_predict = self.sigmoid2(x_predict)

        #返回的第一项是注意力，第二项是预测的输出
        return attention, x_predict

