import os
import numpy as np
import pickle

label_path = '/home/hfutzny/sda/casual_face/dataset/ActionUnit_Labels'
image_path = '/home/hfutzny/sda/casual_face/CIS/data/DISFA/image/'

au_idx = [1, 2, 4, 6, 9, 12, 25, 26]
au_files = os.listdir(label_path)
au_files.sort(key=lambda x:int(x[2:]))  
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
    count_file_path = '/home/hfutzny/sda/casual_face/CIS/label_count/' + au_file + '.txt'
    with open(count_file_path,'w') as cf:
        cf.write('AU1: ')
        cf.write(str(count[0]))
        cf.write('\n')
        cf.write('AU2: ')
        cf.write(str(count[1]))
        cf.write('\n')
        cf.write('AU4: ')
        cf.write(str(count[2]))
        cf.write('\n')
        cf.write('AU6: ')
        cf.write(str(count[3]))
        cf.write('\n')
        cf.write('AU9: ')
        cf.write(str(count[4]))
        cf.write('\n')
        cf.write('AU12: ')
        cf.write(str(count[5]))
        cf.write('\n')
        cf.write('AU25: ')
        cf.write(str(count[6]))
        cf.write('\n')
        cf.write('AU26: ')
        cf.write(str(count[7]))
        cf.write('\n')
        # cf.write('AU1: ' , count[0])
        # cf.write('AU2: ' , count[1])
        # cf.write('AU4: ' , count[2])
        # cf.write('AU6: ' , count[3])
        # cf.write('AU9: ' , count[4])
        # cf.write('AU12: ' , count[5])
        # cf.write('AU25: ' , count[6])
        # cf.write('AU26: ' , count[7])



                




# part1_frame_list = []
# part1_numpy_list = []
# # part1_label_numpy_list = []
# part1_label_list = []

# for fr in part1:


#     imgnums = []   
#     filelist = os.path.join(image_path,fr)  #/home/.../image/SN001
#     files = os.listdir(filelist)
#     files.sort(key=lambda x:int(x[:-4]))  #files = 1.jpg......得到一个列表
#     total_frame = len(files)
#     for filenum in files:
#         imgnums.append(int((filenum.split('.jpg')[0])))

#     fr_path = os.path.join(label_path,fr)

#     au_label_array = np.zeros([total_frame,1,8],dtype=np.int) #8是au_idx是数量 total_frame = 4845 
#     # au_label_array = []
#     for ai, au in enumerate(au_idx):#对所有的au遍历
#         AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
#         if not os.path.isfile(AULabel_path):
#             continue
#         print("--Checking AU:" + str(au) + " ...")
#         t = 0
#         with open(AULabel_path, 'r') as label:
#             for lines in label.readlines():

#                 frameIdx, AUIntensity = lines.split(',')#获得帧的编号和AU强度，上面的t和frameIdx是一样的
#                 frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
#                 if frameIdx in imgnums:
#                     if AUIntensity >= 2:#AU强度大于1表示存在AU
#                         AUIntensity = 1
#                     else:
#                         AUIntensity = 0
#                     au_label_array[t,0,ai] = AUIntensity
#                     t += 1