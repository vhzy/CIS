
import dlib
import face_recognition
import math
import numpy as np
import cv2
import sys
import os
from os.path import basename

 
def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)
 
 
def face_alignment(faces):
    # 预测关键点
    #print("进行对齐-----")
    predictor = dlib.shape_predictor("/home/hfutzny/sda/casual_face/CIS/tests/shape_predictor_68_face_landmarks.dat")
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)
        # left eye, right eye, nose, left mouth, right mouth
        order = [36, 45, 30, 48, 54]
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
        # 计算两眼的中心坐标
        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)
        # 计算角度
        angle = math.atan2(dy, dx) * 180. / math.pi
        # 计算仿射矩阵
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
        # 进行仿射变换，即旋转
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned
 
 
def test(img_path):
    path = "/home/hfutzny/sda/casual_face/CIS/data/DISFA"
    unknown_image = face_recognition.load_image_file(img_path)
    unknown_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)
    # 定位图片中的人脸
    face_locations = face_recognition.face_locations(unknown_image)
    # 提取人脸区域的图片并保存
    src_faces = []
    for rect in face_locations:
        (x, y, w, h) = rect_to_bbox(rect)
        detect_face = unknown_image[y:y+h, x:x+w]
        src_faces.append(detect_face)
        #detect_face = cv2.cvtColor(detect_face, cv2.COLOR_RGBA2BGR)
        #cv2.imwrite("result/face_" + str(src_face_num) + ".jpg", detect_face)
    # 人脸对齐操作并保存
    # len_srcfaces = len(src_faces)
    # print("检测到该图片的人脸数：",len_srcfaces)
    
    faces_aligned = face_alignment(src_faces)
    # len_alignedfaces = len(faces_aligned) 
    # print("对齐的人脸数：",len_alignedfaces)
     
    #判断：如果这张图片没有检测到人脸，将图片的路径记录到日志文本中
    # if len_srcfaces == 0 and len_alignedfaces == 0:
    #     print("该输入图片未检测到人脸，将其写入文本中---")
    #     print(img_path,file=open("fail.txt","a"))
        
  
    fileName = basename(img_path)
    face_num = 0
    for faces in faces_aligned:
        face_num = face_num + 1
        #faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
        resize_image = cv2.resize(faces,(256,256))
        target_path = os.path.join(path,img_path.split('/')[-2])
        os.makedirs(target_path,exist_ok=True)
        cv2.imwrite(os.path.join(target_path,fileName), resize_image)

    pass
 
 
if __name__ == '__main__':
    #单张图片检测
    #image_file = sys.argv[1]
    #test(image_file)
    #批量图片检测，path为你的图片路径

    path = '/home/hfutzny/sda/casual_face/dataset/DISFA'
    filelists = os.listdir(path)
    filelists.sort() #对读取的路径进行排序
    for filelist in filelists:
        files = os.listdir(os.path.join(path,filelist))
        files.sort(key=lambda x:int(x[:-4]))
        #定义一个变量用于计数，你输入的图片个数
        for i,file in enumerate(files):
            img_path = os.path.join(path,filelist,file)
            #print("第{}张图片 {}".format(i,img_path))
            test(img_path)        

    # print(filelist)
    print("---------------OVER------------- !!! ")
    pass




