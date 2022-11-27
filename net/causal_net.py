import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# time model
from net.utils.subject_attention import SubjectAttentionLayer
from net.utils.attention_branch import AttentionBranchLayer
from net.utils.au_embedding import AU_Embedding
# pre-trained backbone
import torchvision.models as models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv_block(c_in, c_out, ks=3, sd=1, batch_norm=True):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(c_in,
                      c_out,
                      kernel_size=ks,
                      stride=sd,
                      padding=(ks - 1) // 2,
                      bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.MaxPool2d(2))
    else:
        return nn.Sequential(
            nn.Conv2d(c_in,
                      c_out,
                      kernel_size=ks,
                      stride=sd,
                      padding=(ks - 1) // 2,
                      bias=True), nn.ReLU(), nn.MaxPool2d(2))


class CAUSAL_NET(nn.Module):
    r"""Baseline

    Args:
        num_class (int): Number of classes for the classification task
        backbone (str): choose from 'simple', 'resnet50', 'resnet101', 'vgg16', 'alexnet'
        hidden_size (int): hidden_size for lstm
        num_layers (int): num_layers for lstm
        num_channels (list): num_channel for tcn
        kernel_size (int): kernel_size for tcn
        batch_norm (bool): for backbone: 'simple' 
        dropout (int): for all the model
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, (T_{in}), C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, num_class)` 
          
    """
    def __init__(self,
                 num_class,
                 backbone='simple',
                 temporal_model='single',
                 hidden_size=256,
                 num_layers=2,
                 num_channels=[512, 256, 256],
                 in_channel = 512,
                 mid_channel=64,
                 kernel_size=2,
                 batch_norm=True,
                 dropout=0.3,
                 subject=False,
                 pooling=False,
                 d_in=512,
                 d_m=256,
                 d_out=512,
                 **kwargs):
        super().__init__()
        assert d_in == d_out
        self.num_class = num_class
        self.in_channel = in_channel
        self.mid_channel = mid_channel

        self.backbone = backbone
        self.temporal_model = temporal_model

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.num_channels = num_channels
        self.kernel_size = kernel_size

        self.batch_norm = batch_norm
        self.dropout = dropout

        self.subject = subject
        self.pooling = pooling

        self.d_in = d_in
        self.d_m = d_m
        self.d_out = d_out

        if self.backbone == 'alexnet':
            self.encoder = nn.Sequential(
                *list(models.alexnet(
                    pretrained=False).children())[0],  # [N, 256, 6, 6]
            )
            self.output_channel = 256
            self.output_size = 6

        elif self.backbone == 'vgg16':
            self.encoder = nn.Sequential(
                *list(models.vgg16(
                    pretrained=False).children())[0],  # [N, 512, 8, 8]
            )
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'squeezenet':
            self.encoder = nn.Sequential(
                *list(models.squeezenet1_0(
                    pretrained=False).children())[0],  # [N, 512, 15, 15]
            )
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'resnet18':
            self.encoder = nn.Sequential(
                *list(models.resnet18(pretrained=False).children())
                [:-1],  # [N, 512, image_size // (2^4), _]
            )
            self.output_channel = 512
            self.output_size = 16


        #注意这里children()下一行要改成[:-2]


        elif self.backbone == 'resnet34':
            self.encoder = nn.Sequential(
                *list(models.resnet34(pretrained=False).children())
                [:-2],  # [N, 512, image_size // (2^4), _]
            )
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'resnet50':
            self.encoder = nn.Sequential(
                *list(models.resnet50(pretrained=False).children())
                [:-1],  # [N, 1024, image_size // (2^4), _]
            )
            self.output_channel = 2048
            self.output_size = 16

        if self.temporal_model == 'single':
            pass
        elif self.temporal_model == 'lstm':
            self.lstm = nn.LSTM(input_size=self.output_channel,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True,
                                dropout=self.dropout)
            self.final = nn.Sequential(
                nn.Linear(self.hidden_size, num_class),
                nn.Sigmoid(),
            )
        #首先计算branch attention的两个输出，in_channel是resnet34最后一层的通道数，mid_channel=64，是中间的过度通道数
        self.branch_attention_net= AttentionBranchLayer(self.in_channel,  self.mid_channel, num_class)
        #下面的Output_size是指resnet的feature_map中的高和宽=output_size
        self.au_embedding = AU_Embedding(self.in_channel, self.mid_channel, self.output_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        

        # subject related
        if self.subject:
            if self.pooling == False:
                self.projector = nn.Linear( #projector
                    self.output_channel * self.output_size * self.output_size,
                    self.d_in)
                self.subject_attention = SubjectAttentionLayer(
                    self.d_in, self.d_m, self.d_out)
                self.final = nn.Sequential(
                    nn.Linear(self.d_out, 64),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(64, num_class),
                    nn.Sigmoid(),
                )
            else:
                self.projector = nn.Linear(self.output_channel, self.d_in)
                self.subject_attention = SubjectAttentionLayer(
                    self.d_in, self.d_m, self.d_out)
                self.final = nn.Sequential(
                    nn.Linear(self.d_out, 64),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(64, num_class),
                    nn.Sigmoid(),
                )
        else:
            if self.pooling == False:
                self.final = nn.Sequential(
                    nn.Linear(
                        self.output_channel * self.output_size *
                        self.output_size, 64),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(64, num_class),
                    nn.Sigmoid(),
                )
            else:
                self.final = nn.Sequential(
                    nn.Linear(self.output_channel, 64),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(64, num_class),
                    nn.Sigmoid(),
                )

    def forward(self, image, subject_infos=None):
        '''
        image for cnn: [N, C, H, W] if single
                        [N, T, C, H, W] if sequential model (time_model is set)
        '''
        N, T, C, H, W = image.shape
        x = image.view(-1, image.shape[2], image.shape[3], image.shape[4])

        x = self.encoder(x)#经过resnet50的输出
        branch_attention, branch_output = self.branch_attention_net(x)
        # print(branch_attention.shape) [4,8,8,8]
        # print(branch_output.shape) [4,8] 这里测试正确，没有问题
        # import sys
        # sys.exit()

        #接下来应该用branch_attention和x做哈达玛积，得到dim[4,8,512,8,8]
        for i in range(self.num_class):
            attention_map1 = branch_attention[:,i,:,:].unsqueeze(1) #[4,1,8,8] x:[4,512,8,8]
            attention_out = attention_map1 * x #[4,512,8,8]
            attention_out = attention_out + x
            #已经得到了某一个AU的表征，继续对其embedding，之后得到这个AU的预测结果[4,1],8个AU就是[4,8]
            au_predict = self.au_embedding(attention_out) #shape[4,1]
            # print(au_predict.shape)  【4，1】
            # import sys
            # sys.exit()
            if i == 0:
                all_au_predict = au_predict
            else:
                all_au_predict = torch.cat([all_au_predict,au_predict],dim = -1) #最后的维度是[4,8]




#以下代码都不要了，但是暂时保存着
        '''
        if self.pooling == False:
            x = x.view(x.shape[0], -1)
        else:

            x = self.avgpool(x)#池化
            x = x.view(x.shape[0], -1)
            #print(x.shape)


        if self.subject:
            x = self.projector(x)
            feature = x
            if subject_infos:
                x = self.subject_attention(x, subject_infos)
        else:
            feature = x
        '''
        if self.temporal_model == 'single':
            pass
        elif self.temporal_model == 'lstm':
            x = x.view(N, -1, self.output_channel)
            h0 = torch.zeros(self.num_layers, x.size(0),
                             self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0),
                             self.hidden_size).to(device)
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x, (h0, c0))  # [N, T, self.hidden_size]
            x = x[:, -1, :]  # N x self.hidden_size

        #output = self.final(x)
        '''
        之前返回的是feature，output，feature是为了在之后为每个subject构建字典，这里我不需要构建字典，所以feature删掉
        增加了一个输出brabch_output，这是branch attention的输出，也用来计算loss
        '''

        return all_au_predict, branch_output