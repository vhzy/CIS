import os
path = '/home/hfut1609/Disk_sda/hzy/faceAU/CIS/data/DISFA/image'      # 输入文件夹地址
files = os.listdir(path)   # 读入文件夹
files.sort(key=lambda x:int(x[2:]))
for file in files:
    file = os.path.join(path,file)
    f = os.listdir(file)
    f.sort
    num_png = len(f)       # 统计文件夹中的文件个数
    print(num_png)             # 打印文件个数
# # 输出所有文件名
# print("所有文件名:")
# for file in files:
#     print(file)
