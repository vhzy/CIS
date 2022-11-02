# 两种方法都能打开
import pickle
import numpy as np

# f = open('/home/hfutzny/sda/casual_face/CIS/data/bp4d_example/val0_label.pkl','rb')
# name, label = pickle.load(f)
# print(label)


np.set_printoptions(threshold=np.inf) #解决显示不完全问题

f = open('/home/hfutzny/sda/casual_face/CIS/data/DISFA/list/train2_label.pkl','rb')
inf = pickle.load(f)
f.close()
print(type(inf))
print(len(inf))
print(inf[0][0])
inf = str(inf)
a = np.array(inf)
print(a.shape)
ft = open("/home/hfutzny/sda/casual_face/CIS/tests/train2_label.txt",'w')
ft.write(inf)



# img_path = './train_data.pkl'
# img_data = np.load(img_path)
# print(img_data)

