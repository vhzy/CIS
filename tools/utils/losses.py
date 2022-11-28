from locale import normalize
import torch
from torch.cuda import device_count
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random
import os
import numpy as np
from scipy import stats

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CLFLoss(nn.Module):
    def __init__(self,
                 weights,
                 class_weights,
                 inner_weights,
                 lambda_clf=1,
                 size_average=True,
                 subject_id = None,
                 **kwargs):
        super(CLFLoss, self).__init__()
        self.size_average = size_average
        self.weights = weights
        self.class_weights = class_weights
        self.inner_weights = inner_weights
        self.lambda_clf = lambda_clf

    def forward(self, outputs, targets, subject_id = None,*args):

        num_class = outputs.size()[-1]
        batch_size = outputs.size()[0]
        outputs = outputs.view(-1, num_class) #[4,8]
        targets = targets.view(-1, num_class)

        N, num_class = outputs.size()
        loss_buff = 0
        
        for i in range(num_class):
            batch_target = targets[:, i] #一个batch里面的第i个AU
            batch_output = outputs[:, i]
            # if self.class_weights is None:
            #     print(1)
            #     import sys
            #     sys.exit()
            #     loss_au = torch.sum(
            #             -(batch_target * torch.log((batch_output + 0.05) / 1.05) +
            #             (1.0 - batch_target) * torch.log((1.05 - batch_output) / 1.05)))
            
            # elif (self.class_weights is not None) and (subject_id is None):
            if subject_id is None:
                # print(2)
                # import sys
                # sys.exit()
                loss_au = torch.sum(-(1 - self.class_weights[i]) * (
                        (1.0 - self.weights[i]) * batch_target * torch.log(
                            (batch_output + 0.05) / 1.05) + self.weights[i] *
                        (1.0 - batch_target) * torch.log((1.05 - batch_output) / 1.05)))
                loss_buff += loss_au
            else:         
                # print(3)
                # import sys
                # sys.exit()
                for j in range(batch_size):
                    id = subject_id[j]
                    target = batch_target[j]
                    output = batch_output[j]
                    loss_au = torch.sum(-(1 - self.class_weights[i]) * (1 - self.inner_weights[id][i]) * (
                        (1.0 - self.weights[i]) * target * torch.log(
                            (output + 0.05) / 1.05) + self.weights[i] *
                        (1.0 - target) * torch.log((1.05 - output) / 1.05)))
                    # print(type(loss_au))
                    # import sys
                    # sys.exit()
                loss_buff += loss_au
        return self.lambda_clf * loss_buff / (num_class * N)


if __name__ == '__main__':
    targets = [[1, 0, 0, 1, 1, 0, 1]]
    outputs = [[0.5, 0.4, 0.1, 0.6, 0.7, 0.2, 0.3]]

    targets = torch.tensor(targets, dtype=torch.float)
    outputs = torch.tensor(outputs, dtype=torch.float)

    loss = CLFLoss()(outputs, targets)