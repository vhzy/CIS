import os
import sys
import bz2
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import cv2

from os.path import basename

# LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
LANDMARKS_MODEL_URL = '/home/hfutzny/sda/casual_face/CIS/tests/shape_predictor_68_face_landmarks.dat'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def draw_landmark_face(img,dlib_68_landmarks,save_name):
    for i,point in enumerate(dlib_68_landmarks):
        # def circle(img, center, radius, color, thickness=None, lineType=None,shift=None):
        cv2.circle(img,center=point,radius=1,color=(0,0,255),thickness=-1)
        cv2.putText(img,"{}".format(i),point,cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255))
        cv2.imwrite(save_name,img)
    return img


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py  ./raw_images ./aligned_images
    """

    # landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
    #                                            LANDMARKS_MODEL_URL, cache_subdir='temp'))

    landmarks_model_path = '/home/hfutzny/sda/casual_face/CIS/tests/shape_predictor_68_face_landmarks.dat'

    # RAW_IMAGES_DIR = sys.argv[1]
    # ALIGNED_IMAGES_DIR = sys.argv[2]



    # ALIGNED_IMAGES_DIR = sys.argv[2]

    ALIGNED_IMAGES_DIR = '/home/hfutzny/sda/casual_face/CIS/data/DISFA'
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    path = '/home/hfutzny/sda/casual_face/dataset/DISFA'
    filelists = os.listdir(path)
    filelists.sort() #对读取的路径进行排序
    for filelist in filelists:
        files = os.listdir(os.path.join(path,filelist))
        files.sort(key=lambda x:int(x[:-4]))
        #定义一个变量用于计数，你输入的图片个数
        for i,img_name in enumerate(files):
            raw_img_path = os.path.join(path,filelist,img_name)
            #print("第{}张图片 {}".format(i,img_path))


    # for img_name in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:
    #     raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)


   

            for j, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                # face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                
                target_path = os.path.join(ALIGNED_IMAGES_DIR,raw_img_path.split('/')[-2])
                os.makedirs(target_path, exist_ok=True)
                aligned_face_path = os.path.join(target_path, basename(raw_img_path))
            
                image_align(raw_img_path, aligned_face_path, face_landmarks,output_size=256)

            # draw = 1
            # if draw:
            #     img = cv2.imread(raw_img_path)
            #     landmark_save_path = os.path.join(ALIGNED_IMAGES_DIR, 'landmarks_' + face_img_name)
            #     draw_landmark_face(img, face_landmarks, save_name=landmark_save_path)


