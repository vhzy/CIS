#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
from torch._C import LockingLogger
import yaml
import numpy as np
import os
import time

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from tools.utils import funcs, losses

from tensorboardX import SummaryWriter
from thop import profile
from torchstat import stat


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def init_environment(self):
        super().init_environment()
        self.best_f1 = np.zeros(self.arg.model_args['num_class'])
        self.best_acc = np.zeros(self.arg.model_args['num_class'])
        self.best_aver_f1 = 0
        self.best_aver_acc = 0
        # self.subject_prototype = dict()
        # self.subject_prototype_update = dict()

    def load_model(self):

        self.train_logger = SummaryWriter(log_dir=os.path.join(
            self.arg.work_dir, 'train'),
                                          comment='train')
        self.validation_logger = SummaryWriter(log_dir=os.path.join(
            self.arg.work_dir, 'validation'),
                                               comment='validation')
        #print(self.arg.model)
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))

        update_dict = {}
        model_dict = self.model.state_dict()
        # self.io.print_log(str(self.model.encoder))
        # import sys
        # sys.exit()   
        if self.arg.pretrain and self.arg.model_args['backbone'] in [
                'resnet18'
        ]:
            pretrained_dict = models.resnet18(weights='ResNet18_Weights.DEFAULT').state_dict()
            for k, v in pretrained_dict.items():
                if "layer1" in k:
                    update_dict[k.replace("layer1", "encoder.4", 1)] = v
                elif "layer2" in k:
                    update_dict[k.replace("layer2", "encoder.5", 1)] = v
                elif "layer3" in k:
                    update_dict[k.replace("layer3", "encoder.6", 1)] = v
                elif "layer4" in k and self.arg.model_args[
                        'backbone'] == 'resnet18':
                    update_dict[k.replace("layer4", "encoder.7", 1)] = v
                elif k == 'conv1.weight':
                    update_dict['encoder.0.weight'] = v
                elif k == 'bn1.weight':
                    update_dict['encoder.1.weight'] = v
                elif k == 'bn1.bias':
                    update_dict['encoder.1.bias'] = v
                elif k == 'bn1.running_mean':
                    update_dict['encoder.1.running_mean'] = v
                elif k == 'bn1.running_var':
                    update_dict['encoder.1.running_var'] = v
                elif k == 'bn1.num_batches_tracked':
                    update_dict['encoder.1.num_batches_tracked'] = v
        elif self.arg.pretrain and self.arg.model_args['backbone'] in [
                'resnet34'
        ]:
            pretrained_dict = models.resnet34(weights='ResNet34_Weights.DEFAULT').state_dict()
            # self.io.print_log(str(pretrained_dict.keys()))
            # import sys
            # sys.exit()
            for k, v in pretrained_dict.items():
                if "layer1" in k:
                    update_dict[k.replace("layer1", "encoder.4", 1)] = v
                elif "layer2" in k:
                    update_dict[k.replace("layer2", "encoder.5", 1)] = v
                elif "layer3" in k:
                    update_dict[k.replace("layer3", "encoder.6", 1)] = v
                elif "layer4" in k and self.arg.model_args[
                        'backbone'] == 'resnet34':
                    update_dict[k.replace("layer4", "encoder.7", 1)] = v
                elif k == 'conv1.weight':
                    update_dict['encoder.0.weight'] = v
                elif k == 'bn1.weight':
                    update_dict['encoder.1.weight'] = v
                elif k == 'bn1.bias':
                    update_dict['encoder.1.bias'] = v
                elif k == 'bn1.running_mean':
                    update_dict['encoder.1.running_mean'] = v
                elif k == 'bn1.running_var':
                    update_dict['encoder.1.running_var'] = v
                elif k == 'bn1.num_batches_tracked':
                    update_dict['encoder.1.num_batches_tracked'] = v
        elif self.arg.pretrain and self.arg.model_args['backbone'] in [
                'resnet50'
        ]:
            pretrained_dict = models.resnet50(weights='ResNet50_Weights.DEFAULT').state_dict()
            for k, v in pretrained_dict.items():
                if "layer1" in k:
                    update_dict[k.replace("layer1", "encoder.4", 1)] = v
                elif "layer2" in k:
                    update_dict[k.replace("layer2", "encoder.5", 1)] = v
                elif "layer3" in k:
                    update_dict[k.replace("layer3", "encoder.6", 1)] = v
                elif "layer4" in k and self.arg.model_args[
                        'backbone'] == 'resnet50':
                    update_dict[k.replace("layer4", "encoder.7", 1)] = v
                elif k == 'conv1.weight':
                    update_dict['encoder.0.weight'] = v
                elif k == 'bn1.weight':
                    update_dict['encoder.1.weight'] = v
                elif k == 'bn1.bias':
                    update_dict['encoder.1.bias'] = v
                elif k == 'bn1.running_mean':
                    update_dict['encoder.1.running_mean'] = v
                elif k == 'bn1.running_var':
                    update_dict['encoder.1.running_var'] = v
                elif k == 'bn1.num_batches_tracked':
                    update_dict['encoder.1.num_batches_tracked'] = v
        elif self.arg.pretrain and self.arg.model_args['backbone'] in [
                'vgg16'
        ]:
            pretrained_dict = models.vgg16(pretrained=True).state_dict()
            for k, v in pretrained_dict.items():
                if "features" in k:
                    update_dict[k.replace("features", "encoder", 1)] = v
        elif self.arg.pretrain and self.arg.model_args['backbone'] in [
                'squeezenet'
        ]:
            pretrained_dict = models.squeezenet1_0(
                pretrained=True).state_dict()
            for k, v in pretrained_dict.items():
                if "features" in k:
                    update_dict[k.replace("features", "encoder", 1)] = v
        elif self.arg.pretrain and self.arg.model_args['backbone'] in [
                'alexnet'
        ]:
            pretrained_dict = models.alexnet(pretrained=True).state_dict()
            for k, v in pretrained_dict.items():
                if "features" in k:
                    update_dict[k.replace("features", "encoder", 1)] = v

        # self.io.print_log(str(update_dict.keys()))
        # import sys
        # sys.exit()            
        print('updated params:{}'.format(len(update_dict)))
        model_dict.update(update_dict)
        self.model.load_state_dict(model_dict)

        if self.arg.loss == 'clf':
            inner_weight = np.array(self.arg.loss_inner_weight)
            inner_weight = torch.from_numpy(inner_weight).float()

            class_weight = np.array(self.arg.loss_class_weight)  #类间的标签权重

            class_weight = torch.from_numpy(class_weight).float()
            weight = np.array(self.arg.loss_weight)
            weight = torch.from_numpy(weight).float()
            self.loss = losses.CLFLoss(weights=weight,class_weights = class_weight, inner_weights = inner_weight)
        else:
            raise ValueError()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.arg.base_lr,
                                       momentum=0.9,
                                       nesterov=self.arg.nesterov,
                                       weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.arg.base_lr,
                                        weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(),
                                           lr=self.arg.base_lr,
                                           alpha=0.9,
                                           momentum=0,
                                           weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (self.arg.lr_decay**np.sum(
                self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        branch_result_frag = []
        result_frag = []
        label_frag = []
        subject_ID = []

        print('training dataloder length: ', len(loader))

        for image, label, subject_id in loader:
            subject_map = {
            'SN001': 0,'SN002': 1,'SN003': 2,'SN004': 3,'SN005': 4,'SN006': 5,'SN007': 6,
            'SN008': 7,'SN009': 8,'SN010': 9,'SN011': 10,'SN012': 11,'SN013': 12,'SN016': 13,'SN017': 14, 'SN018': 15,
            'SN021': 16,'SN023': 17,'SN024': 18,'SN025': 19,'SN026': 20,'SN027': 21,'SN028': 22,'SN029': 23,
            'SN030': 24,'SN031': 25,'SN032': 26,}

            # print(subject_id)
            for i in range(len(subject_id)):
                subject_ID.append(subject_map[subject_id[i]])
            # print(subject_id)
            # print(subject_ID)
            # import sys
            # sys.exit()

            # get data
            label = label.float().to(self.dev)
            image = image.float().to(self.dev)

            # forward 对于大于0epoch时，已经计算出来了字典，可以进行注意力
            # if self.meta_info['epoch'] >= self.arg.clf_only_epoch:
            #     feature, output = self.model(
            #         image, subject_infos=[self.K_subject, self.V_subject])
            # else:
            #     feature, output = self.model(image)
            output, branch_output = self.model(image)

            # collect subject feature，这里是创建每个人的混淆字典特征
            # for sidx, sub_id in enumerate(subject_id):
            #     if sub_id not in self.subject_prototype_update:
            #         self.subject_prototype_update[sub_id] = []
            #         self.subject_prototype_update[sub_id].append(
            #             feature[sidx].squeeze().data.cpu().numpy())
            #     else:
            #         self.subject_prototype_update[sub_id].append(
            #             feature[sidx].squeeze().data.cpu().numpy())

            result_frag.append(output.data.cpu().numpy())
            branch_result_frag.append(branch_output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())

            # if self.meta_info['epoch'] > 0 or self.arg.clf_only_epoch > 1:
            #     loss = self.loss(output, label)
            #     # backward，大于0时进行梯度回传，否则第一轮式停止反向传播来初始化词典
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.optimizer.step()
            # else:
            #     loss = torch.tensor(0)
            loss = self.loss(output, label, subject_ID) + self.arg.branch_loss_weight * self.loss(branch_output, label) #加上分支
            # loss = self.arg.branch_loss_weight * self.loss(branch_output, label)
            # print(type(loss))
            # import sys
            # sys.exit()
            # backward，大于0时进行梯度回传，否则第一轮式停止反向传播来初始化词典
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

        # update subject_prototype
        '''
        for k, v in self.subject_prototype_update.items():
            self.subject_prototype[k] = np.stack(
                self.subject_prototype_update[k])

        items = sorted(self.subject_prototype.items(), key=lambda d: d[0])
        for item in items:
            self.subject_prototype[item[0]] = item[1]

        N_subject = len(self.subject_prototype.keys())
        total_samples = np.sum([
            len(self.subject_prototype[k])
            for k in self.subject_prototype.keys()
        ])

        self.P_subject = torch.zeros(N_subject).to(image.device)
        self.K_subject = torch.zeros(
            (N_subject, feature.shape[-1])).to(image.device)
        self.V_subject = torch.zeros(
            (N_subject, feature.shape[-1])).to(image.device)
        for idx, (k, v) in enumerate(self.subject_prototype.items()):
            self.P_subject[idx] = len(
                self.subject_prototype[k]) / total_samples
            self.K_subject[idx] = torch.tensor(np.mean(v, axis=0))
            self.V_subject[idx] = self.P_subject[idx] * self.K_subject[idx]

        self.subject_prototype_update = dict()
        '''

        # visualize loss and metrics
        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)
        _, _, train_f1, train_acc = funcs.record_metrics(
            self.result, self.label, self.epoch_info['mean_loss'],
            self.arg.model_args['num_class'], self.arg.work_dir, 'train')
        self.train_logger.add_scalar('loss', self.epoch_info['mean_loss'],
                                     self.meta_info['epoch'])
        self.train_logger.add_scalar('train-acc', train_acc,
                                     self.meta_info['epoch'])
        self.train_logger.add_scalar('train-F1', train_f1,
                                     self.meta_info['epoch'])

        state = {
            'model': self.model.state_dict(),
            # 'K_subject': self.K_subject,
            # 'V_subject': self.V_subject,
            # 'P_subject': self.P_subject
        }
        torch.save(state, os.path.join(self.arg.work_dir, 'final_model.pt'))

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        result_frag = []
        loss_value = []
        label_frag = []

        print('validation dataloder length: ', len(loader))

        for image, label, subject_id in loader:

            # get data
            label = label.float().to(self.dev)
            image = image.float().to(self.dev)

            # inference
            with torch.no_grad():
                # if self.meta_info['epoch'] >= self.arg.clf_only_epoch:
                #     _, output = self.model(
                #         image, subject_infos=[self.K_subject, self.V_subject])
                # else:
                #     _, output = self.model(image)
                _, output = self.model(image)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()
            # compute f1 score
            f1_score, accuracy, val_f1, val_acc = funcs.record_metrics(
                self.result, self.label, self.epoch_info['mean_loss'],
                self.arg.model_args['num_class'], self.arg.work_dir, 'val')
            if self.best_aver_f1 < val_f1 :
            # and self.meta_info['epoch'] >= self.arg.clf_only_epoch:
                self.best_aver_f1 = val_f1
                self.best_f1 = f1_score
                self.best_aver_acc = val_acc
                self.best_acc = accuracy

            self.validation_logger.add_scalar('loss',
                                              self.epoch_info['mean_loss'],
                                              self.meta_info['epoch'])
            self.validation_logger.add_scalar('val-acc', val_acc,
                                              self.meta_info['epoch'])
            self.validation_logger.add_scalar('val-F1', val_f1,
                                              self.meta_info['epoch'])

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(add_help=add_help,
                                         parents=[parent_parser],
                                         description='causal-net')

        # region arguments yapf: disable
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--lr_decay', type=float, default=0.1, help='lr decay for optimizer')
        parser.add_argument('--loss', type=str, default='Focal', help='loss for optimizer')
        parser.add_argument('--loss_weight', type=int, default=[], nargs='+', help='weights for BCE loss')
        parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
        #parser.add_argument('--backbone_only', type=str2bool, default=True, help='only use backbone weights')
        parser.add_argument('--pretrain', type=str2bool, default=True, help='load pretrained weights on ImageNet or not')
        parser.add_argument('--clf_only_epoch', type=int, default=1, help='clf only epoch')
        parser.add_argument('--branch_loss_weight', type=float, default=0.33, help='weight of branch loss')
        parser.add_argument('--loss_class_weight', type=float, default=[], nargs='+', help='class weights for BCE loss')
        parser.add_argument('--loss_inner_weight', type=float, default=[], nargs='+', help='class inner weights for BCE loss')
        # endregion yapf: enable

        return parser
