#run-cisnet.py中使用
import os
import json
import numpy as np
import pickle
import argparse
import random
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def compute_frequency(args, infodir):
    sample_names = [
        p for p in os.listdir(infodir)
        if os.path.isfile(os.path.join(infodir, p)) and 'json' in p
    ]

    if 'BP4D' in args.datasetdir:
        subject_names = np.unique(
            [sample_name[0:4] for sample_name in sample_names]).tolist()
    elif 'DISFA' in args.datasetdir:
        subject_names = np.unique(
            [sample_name[0:5] for sample_name in sample_names]).tolist()

    namelist = []
    for name in namelist:
        subject_names.remove(name)

    if 'BP4D' in args.datasetdir:
        # BP4D
        aulist_selected = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
        au_map = {
            1: 0,
            2: 1,
            4: 2,
            6: 4,
            7: 5,
            10: 7,
            12: 9,
            14: 11,
            15: 12,
            17: 14,
            23: 19,
            24: 20
        }
    elif 'DISFA' in args.datasetdir:
        # DISFA
        aulist_selected = [1, 2, 4, 6, 9, 12, 25, 26]
        au_map = {1: 0, 2: 1, 4: 2, 6: 4, 9: 5, 12: 6, 25: 10, 26: 11}

    cntsample = 0
    all_labelcnt = list()
    subject_labelcnt = np.zeros((len(subject_names), len(aulist_selected)))
    for sample_name in sample_names:
        sample_path = os.path.join(infodir, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        frame_infos = video_info['data']
        frame_infos = sorted(frame_infos,
                             key=lambda e: e['frame_index'],
                             reverse=False)

        cntframe = 0
        labelcnt = np.zeros(len(aulist_selected))
        for frame_info in frame_infos:

            label = list()
            for au in aulist_selected:
                label.append(int(frame_info['label'][au_map[au]]))
            label = np.array(label)

            labelcnt += label
            cntframe += 1
        all_labelcnt.append(labelcnt.tolist())
        for idx, subject in enumerate(subject_names):
            if subject in sample_name:
                subject_labelcnt[idx] += labelcnt
        cntsample += 1

    all_labelcnt = np.array(all_labelcnt)

    return subject_labelcnt, subject_names

#计算所有AU里面 positive的ratio
def compute_label_frequency(label_path):
    # load label
    with open(label_path, 'rb') as f:
         labels = pickle.load(f)
    f.close()
    labels = np.concatenate(labels, axis=0)
    all_labelcnt = np.sum(labels, axis=0)

    return all_labelcnt / labels.shape[0]


def compute_class_frequency(label_path):
    # load label
    with open(label_path, 'rb') as f:
         labels = pickle.load(f)
    f.close()
    labels = np.concatenate(labels, axis=0)
    all_labelcnt = np.sum(labels, axis=0)

    return all_labelcnt / all_labelcnt.sum()

def compute_AU_inner_frequency(label_path):
    # au_list = [] #au_list存储所有的au1...au26,每个au26里面有每个受试者的权重
    # au_list_part = []

    # au_idx = [1, 2, 4, 6, 9, 12, 25, 26]
    # files_list = os.listdir(label_path)
    # files_list.sort(key=lambda x:int(x[2:]))  #SN001
    # for files in files_list:
    #     SN_path = os.path.join(label_path, files)
    #     file = os.listdir(SN_path)
    #     file.sort(key=lambda x:int(x.split('.')[0][8:]))
    #     for i in file:
    #         if i.split('.')[0][8:] in au_idx:
    #             with open(os.path.join(SN_path,i), 'rb') as f:
    #                 for lines in f.readlines():
    #                     frameIdx, AUIntensity = lines.split(',')#获得帧的编号和AU强度，上面的t和frameIdx是一样的
    #                     frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
    #                     if AUIntensity >= 2:#AU强度大于1表示存在AU
    #                         AUIntensity = 1
    #                     else:
    #                         AUIntensity = 0

    au_idx = [1, 2, 4, 6, 9, 12, 25, 26]
    au_files = os.listdir(label_path)
    au_files.sort(key=lambda x:int(x[2:]))  
    au_all = []
    au_res = []
    for au_file in au_files:
        count = [0,0,0,0,0,0,0,0]
        for ai, au in enumerate(au_idx): #得到每个subject的标签SN001_au1.txt
            l_path = os.path.join(label_path , au_file) 
            AULabel_path = os.path.join(l_path,au_file+'_au'+str(au) +'.txt')
            if not os.path.isfile(AULabel_path):
                continue
            with open(AULabel_path, 'r') as label:
                for lines in label.readlines():
                    frameIdx, AUIntensity = lines.split(',')#获得帧的编号和AU强度，上面的t和frameIdx是一样的
                    frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                    if AUIntensity >= 2:#AU强度大于1表示存在AU
                        count[ai] += 1
        au_all.append(count)
    au_all = np.array(au_all)
    au_class_freq = np.sum(au_all,axis=0)
    #print(au_class_freq)
    for i in range(len(au_all)):  #第i个subject的第j个AU
        sub_freq = []
        for j in range(len(au_all[i])):
            freq = au_all[i][j] / au_class_freq[j]
            sub_freq.append(freq)
        au_res.append(sub_freq)
    au_res = np.array(au_res)
    return au_res



    

# if __name__ == '__main__':
#     label_path = '/home/hfutzny/sda/casual_face/CIS/data/ActionUnit_Labels'
#     compute_AU_inner_frequency(label_path)

def split_data_random(args, infodir, kfold, splits_index):

    subject_labelcnt, subject_names = compute_frequency(args, infodir)
    subject_labelcnt = subject_labelcnt.tolist()
    idxs = list(zip(subject_labelcnt, subject_names))
    random.shuffle(idxs)
    subject_labelcnt[:], subject_names[:] = zip(*idxs)
    subject_labelcnt = np.array(subject_labelcnt)

    splits_labelcnt = np.zeros((kfold, subject_labelcnt.shape[1]))
    splits = -1 * np.ones((kfold, subject_labelcnt.shape[0]))
    cnts = np.zeros(kfold)

    for i in range(subject_labelcnt.shape[0]):
        kidx = i % kfold
        splits_labelcnt[kidx] += subject_labelcnt[i]
        splits[kidx][int(cnts[kidx])] = i
        cnts[kidx] += 1

    print("splits labelcnt:\n", splits_labelcnt)

    with open(
            os.path.join(args.datasetdir,
                         "splits_r" + str(splits_index) + ".txt"), 'w') as f:
        for i in range(kfold):
            cnt = 0
            while splits[i][cnt] != -1:
                f.write(subject_names[int(splits[i][cnt])] + ' ')
                cnt += 1
                if cnt == subject_labelcnt.shape[0]:
                    break
            f.write('\n')
    f.close()

    return