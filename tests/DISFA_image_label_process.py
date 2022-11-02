import os
import numpy as np
import pickle
#You nead downloading DISFA including 'ActionUnit_Labels'
label_path = '/home/hfutzny/sda/casual_face/dataset/ActionUnit_Labels'
list_path_prefix = '/home/hfutzny/sda/casual_face/CIS/data/DISFA/list/'
image_path = '/home/hfutzny/sda/casual_face/CIS/data/DISFA/image/'

part1 = ['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN009','SN016']
part2 = ['SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024']
part3 = ['SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004']

# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

au_idx = [1, 2, 4, 6, 9, 12, 25, 26]
#au_idx = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

with open(list_path_prefix + 'test2_imagepath.pkl','wb') as f:
    u = 0
with open(list_path_prefix + 'test2_label.pkl', 'wb') as f1:
    u = 0
part1_frame_list = []
part1_numpy_list = []
for fr in part1:
    fr_path = os.path.join(label_path,fr)
    au1_path = os.path.join(fr_path,fr+'_au1.txt')
    with open(au1_path, 'r') as label:
        total_frame = len(label.readlines())
    if fr == 'SN010':
        total_frame = 4844
    au_label_array = np.zeros([total_frame,1,8],dtype=np.int) #12是au_idx是数量 total_frame = 4845 
    # au_label_array = []
    for ai, au in enumerate(au_idx):#对所有的au遍历
        AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
        if not os.path.isfile(AULabel_path):
            continue
        print("--Checking AU:" + str(au) + " ...")
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                
                frameIdx, AUIntensity = lines.split(',')#获得帧的编号和AU强度，上面的t和frameIdx是一样的
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if AUIntensity >= 2:#AU强度大于1表示存在AU
                    AUIntensity = 1
                else:
                    AUIntensity = 0
                au_label_array[t,0,ai] = AUIntensity
    part1_numpy_list = au_label_array
    part1_label_list = au_label_array.tolist()
            #     au_label_array.append(AUIntensity)
            # au_label_arrays = []
            # au_label_arrays.append(au_label_array)

    # part1_numpy_list.append(au_label_array_list)
    for i in range(total_frame):
        frame_img_name = image_path + fr + '/' + str(i+1) + '.jpg'   #SN001  / 1 .PNG,这里前面的路径也要加上
        frame_img_names=[]
        frame_img_names.append(frame_img_name)
        part1_frame_list.append(frame_img_names)
    with open(list_path_prefix + 'test2_imagepath.pkl', 'wb') as f:
        pickle.dump(part1_frame_list,f)

#     part1_numpy_list.append(au_label_array)
# part1_numpy_list = np.concatenate(part1_numpy_list,axis=0)

with open(list_path_prefix + 'test2_label.pkl', 'wb') as f1:
    pickle.dump(part1_label_list,f1)
# part1_numpy_list = np.concatenate(part1_numpy_list,axis=0)
# part1 test for fold3
# np.savetxt(list_path_prefix + 'DISFA_test_label_fold3.txt', part1_numpy_list,fmt='%d', delimiter=' ')

#################################################################################
with open(list_path_prefix + 'test1_imagepath.pkl','wb') as f:
    u = 0
with open(list_path_prefix + 'test1_label.pkl', 'wb') as f1:
    u = 0
part2_frame_list = []
part2_numpy_list = []
for fr in part2:
    fr_path = os.path.join(label_path,fr)
    au1_path = os.path.join(fr_path,fr+'_au1.txt')
    with open(au1_path, 'r') as label:
        total_frame = len(label.readlines())
    if fr == 'SN010':
        total_frame = 4844
    au_label_array = np.zeros([total_frame,1,8],dtype=np.int) #12是au_idx是数量 total_frame = 4845 
    # au_label_array = []
    for ai, au in enumerate(au_idx):#对所有的au遍历
        AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
        if not os.path.isfile(AULabel_path):
            continue
        print("--Checking AU:" + str(au) + " ...")
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                
                frameIdx, AUIntensity = lines.split(',')#获得帧的编号和AU强度，上面的t和frameIdx是一样的
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if AUIntensity >= 2:#AU强度大于1表示存在AU
                    AUIntensity = 1
                else:
                    AUIntensity = 0
                au_label_array[t,0,ai] = AUIntensity
    part2_numpy_list = au_label_array
    part2_label_list = au_label_array.tolist()

            #     au_label_array.append(AUIntensity)
            # au_label_arrays = []
            # au_label_arrays.append(au_label_array)

    # part1_numpy_list.append(au_label_array_list)
    for i in range(total_frame):
        frame_img_name = image_path + fr + '/' + str(i+1) + '.jpg'   #SN001  / 1 .PNG,这里前面的路径也要加上
        frame_img_names=[]
        frame_img_names.append(frame_img_name)
        part2_frame_list.append(frame_img_names)
    with open(list_path_prefix + 'test1_imagepath.pkl', 'wb') as f:
        pickle.dump(part2_frame_list,f)

#     part1_numpy_list.append(au_label_array)
# part1_numpy_list = np.concatenate(part1_numpy_list,axis=0)

with open(list_path_prefix + 'test1_label.pkl', 'wb') as f1:
    pickle.dump(part2_label_list,f1)

#################################################################################
with open(list_path_prefix + 'test0_imagepath.pkl','wb') as f:
    u = 0
with open(list_path_prefix + 'test0_label.pkl', 'wb') as f1:
    u = 0
part3_frame_list = []
part3_numpy_list = []
for fr in part3:
    fr_path = os.path.join(label_path,fr)
    au1_path = os.path.join(fr_path,fr+'_au1.txt')
    with open(au1_path, 'r') as label:
        total_frame = len(label.readlines())
    if fr == 'SN010':
        total_frame = 4844
    au_label_array = np.zeros([total_frame,1,8],dtype=np.int) #12是au_idx是数量 total_frame = 4845 
    # au_label_array = []
    for ai, au in enumerate(au_idx):#对所有的au遍历
        AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
        if not os.path.isfile(AULabel_path):
            continue
        print("--Checking AU:" + str(au) + " ...")
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                
                frameIdx, AUIntensity = lines.split(',')#获得帧的编号和AU强度，上面的t和frameIdx是一样的
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if AUIntensity >= 2:#AU强度大于1表示存在AU
                    AUIntensity = 1
                else:
                    AUIntensity = 0
                au_label_array[t,0,ai] = AUIntensity
    part3_numpy_list = au_label_array
    part3_label_list = au_label_array.tolist()

            #     au_label_array.append(AUIntensity)
            # au_label_arrays = []
            # au_label_arrays.append(au_label_array)

    # part1_numpy_list.append(au_label_array_list)
    for i in range(total_frame):
        frame_img_name = image_path + fr + '/' + str(i+1) + '.jpg'   #SN001  / 1 .PNG,这里前面的路径也要加上
        frame_img_names=[]
        frame_img_names.append(frame_img_name)
        part3_frame_list.append(frame_img_names)
    with open(list_path_prefix + 'test0_imagepath.pkl', 'wb') as f:
        pickle.dump(part3_frame_list,f)

#     part1_numpy_list.append(au_label_array)
# part1_numpy_list = np.concatenate(part1_numpy_list,axis=0)

with open(list_path_prefix + 'test0_label.pkl', 'wb') as f1:
    pickle.dump(part3_label_list,f1)

#################################################################################
with open(list_path_prefix + 'train0_imagepath.pkl','wb') as f:
    u = 0
with open(list_path_prefix + 'train0_label.pkl','wb') as f:
    u = 0
train_img_label_fold1_list = part1_frame_list + part2_frame_list
with open(list_path_prefix + 'train0_imagepath.pkl', 'wb') as f:
    pickle.dump(train_img_label_fold1_list,f)
# for frame_img_name in train_img_label_fold1_list:
# 	with open(list_path_prefix + 'DISFA_train_img_path_fold1.txt', 'a+') as f:
# 		f.write(frame_img_name + '\n')
train_img_label_fold1_numpy_list = np.concatenate((part1_numpy_list, part2_numpy_list), axis=0)
train_img_label_fold1_list = train_img_label_fold1_numpy_list.tolist()
with open(list_path_prefix + 'train0_label.pkl', 'wb') as f1:
    pickle.dump(train_img_label_fold1_list,f1)
# np.savetxt(list_path_prefix + 'DISFA_train_label_fold1.txt', train_img_label_fold1_numpy_list, fmt='%d')

#################################################################################
with open(list_path_prefix + 'train1_imagepath.pkl','wb') as f:
    u = 0
with open(list_path_prefix + 'train1_label.pkl','wb') as f:
    u = 0
train_img_label_fold2_list = part1_frame_list + part3_frame_list
with open(list_path_prefix + 'train1_imagepath.pkl', 'wb') as f:
    pickle.dump(train_img_label_fold2_list,f)
train_img_label_fold2_numpy_list = np.concatenate((part1_numpy_list, part3_numpy_list), axis=0)
train_img_label_fold2_list = train_img_label_fold2_numpy_list.tolist()
with open(list_path_prefix + 'train1_label.pkl', 'wb') as f1:
    pickle.dump(train_img_label_fold2_list,f1)

#################################################################################
with open(list_path_prefix + 'train2_imagepath.pkl','wb') as f:
    u = 0
with open(list_path_prefix + 'train2_label.pkl','wb') as f:
    u = 0
train_img_label_fold3_list = part2_frame_list + part3_frame_list
with open(list_path_prefix + 'train2_imagepath.pkl', 'wb') as f:
    pickle.dump(train_img_label_fold3_list,f)
train_img_label_fold3_numpy_list = np.concatenate((part2_numpy_list, part3_numpy_list), axis=0)
train_img_label_fold3_list = train_img_label_fold3_numpy_list.tolist()
with open(list_path_prefix + 'train2_label.pkl', 'wb') as f1:
    pickle.dump(train_img_label_fold3_list,f1)