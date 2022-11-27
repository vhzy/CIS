from sys import flags
import numpy as np
import os
import argparse
from ruamel import yaml
from torch.autograd.grad_mode import F
from au_lib.data_utils import compute_label_frequency

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--processor_name',
                        type=str,
                        default='train-causal-net',
                        help='processor name')
    parser.add_argument('-c',
                        '--config_dir',
                        type=str,
                        default='./config/exp1',
                        help='config dir name')
    parser.add_argument('-w',
                        '--work_dir',
                        type=str,
                        default='./work_dir/train/disfa/exp1',
                        help='work dir name')
    parser.add_argument('-d',
                        '--data_dir',
                        type=str,
                        default='/home/hfut1609/Disk_sda/hzy/faceAU/CIS/data/DISFA/list_random3',
                        help='data dir name')
    parser.add_argument('-k', '--kfold', type=int, default=3, help='kfold')
    parser.add_argument('--num_class',
                        type=int,
                        default=8,
                        help='num of class to detect')

    args = parser.parse_args()

    if not os.path.exists(args.config_dir):
        os.mkdir(args.config_dir)

    for k in range(args.kfold):

        label_freq = compute_label_frequency(
            os.path.join(args.data_dir, 'train' + str(k) + '_label.pkl'))
        
        label_class_freq = compute_class_frequency(os.path.join(args.data_dir, 'train' + str(k) + '_label.pkl'))

        desired_caps = {
            'work_dir': os.path.join(args.work_dir, str(k)),
            'feeder': 'feeder.feeder_image_causal.Feeder',
            'train_feeder_args': {
                'label_path':
                os.path.join(args.data_dir, 'train' + str(k) + '_label.pkl'),
                'image_path':
                os.path.join(args.data_dir,
                             'train' + str(k) + '_imagepath.pkl'),
                'image_size':
                256,
                'istrain':
                True,
            },
            'test_feeder_args': {
                'label_path':
                os.path.join(args.data_dir, 'test' + str(k) + '_label.pkl'),
                'image_path':
                os.path.join(args.data_dir, 'test' + str(k) + '_imagepath.pkl'),
                'image_size':
                256,
                'istrain':
                False,
            },
            'model': 'net.causal_net.CAUSAL_NET',
            'model_args': {
                'num_class': args.num_class,
                'backbone': 'resnet34',
                'temporal_model': 'single',
                'subject': True,
                'pooling': True,
                'd_in': 512,
                'd_m': 256,
                'd_out': 512,
            },
            'log_interval': 1000,
            'save_interval': 5,
            'device': [0],
            'batch_size': 4,
            'test_batch_size': 4,
            'base_lr': 0.001,
            'lr_decay': 0.3,
            'step': [],
            'num_epoch': 20,
            'debug': False,
            'num_worker': 0,
            'optimizer': 'SGD',
            'weight_decay': 0.0005,
            'loss': 'clf',
            'loss_weight': label_freq.tolist(),
            'pretrain': True,
            'seed': 42,
            'loss_class_weight': label_class_freq.tolist(),
        }

        yamlpath = os.path.join(args.config_dir, 'train' + str(k) + '.yaml')
        with open(yamlpath, "w", encoding="utf-8") as f:
            yaml.dump(desired_caps, f, Dumper=yaml.RoundTripDumper)

        #main函数里面使用了subparser,所以在terminal使用python进行程序的运行的时候先要指定子解释器的名字，再输入参数。
        #python main.py train-image-casual -c yamlpath
        cmdline = "python main.py " + args.processor_name + " -c " + yamlpath
        print(cmdline)
        os.system(cmdline)
