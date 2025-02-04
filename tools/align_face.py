import math
import cv2
import PIL
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow
import os
import face_recognition
from collections import defaultdict
import numpy as np
from os.path import basename


def detect_landmark(image_array, model_type="large"):
    """ return landmarks of a given image array
    :param image_array: numpy array of a single image
    :param model_type: 'large' returns 68 landmarks; 'small' return 5 landmarks
    :return: dict of landmarks for facial parts as keys and tuple of coordinates as values
    """
    face_landmarks_list = face_recognition.face_landmarks(image_array, model=model_type)
    face_landmarks_list = face_landmarks_list[0]
    return face_landmarks_list


def align_face(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle


def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks


def corp_face(image_array, landmarks, crop_size = 256):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                   np.array(landmarks['bottom_lip'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part 

    w = h = bottom - top
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = cropped_img.resize((crop_size, crop_size),Image.ANTIALIAS)
    cropped_img = np.array(cropped_img)
    return cropped_img


def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (landmark[0] - left, landmark[1] - top)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks


def face_process(image, landmark_model_type='large',crop_size=256):
    """ for a given image, do face alignment and crop face
    :param image: numpy array of a single image
    :param landmark_model_type: 'large' returns 68 landmarks; 'small' return 5 landmarks
    :return:
    cropped_face: image array with face aligned and cropped
    transferred_landmarks: landmarks that fit cropped_face
    """
    # detect landmarks
    face_landmarks_dict = detect_landmark(image_array=image, model_type=landmark_model_type)
    # rotate image array to align face
    aligned_face, eye_center, angle = align_face(image_array=image, landmarks=face_landmarks_dict)
    # rotate landmarks coordinates to fit the aligned face
    rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict,
                                         eye_center=eye_center, angle=angle, row=image.shape[0])
    # crop face according to landmarks
    cropped_face = corp_face(image_array=aligned_face, landmarks=rotated_landmarks, crop_size = crop_size)
    # transfer landmarks to fit the cropped face
    #cropped_face = cropped_face.resize(crop_size)
    return cropped_face


def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature])
    imshow(origin_img)


if __name__ == '__main__':
    raw_img_path = '/home/hfut1609/Disk_sda/hzy/faceAU/CIS/data/DISFA/img/SN001/3233.jpg'
    img = cv2.imread(raw_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = face_process(image=img,landmark_model_type='large',crop_size=256)
    face = Image.fromarray(face)
    face.save('/home/hfut1609/Disk_sda/hzy/faceAU/CIS/tests/img2.jpg', 'JPEG')
    # ALIGNED_IMAGES_DIR = '/home/hfut1609/Disk_sda/hzy/faceAU/CIS/data/DISFA/img'
    # path = '/home/hfut1609/Disk_sda/hzy/faceAU/dataset/raw_images'
    # filelists = os.listdir(path)
    # filelists.sort() #对读取的路径进行排序
    # for filelist in filelists:
    #     files = os.listdir(os.path.join(path,filelist))
    #     files.sort(key=lambda x:int(x[:-4]))
    #     #定义一个变量用于计数，你输入的图片个数
    #     for i,img_name in enumerate(files):
    #         raw_img_path = os.path.join(path,filelist,img_name)
    #         # load image

    #         img = cv2.imread(raw_img_path)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #         # process the face image
    #         face = face_process(image=img,landmark_model_type='large',crop_size=256)
    #         #print(type(face))
    #         face = Image.fromarray(face)
    #         target_path = os.path.join(ALIGNED_IMAGES_DIR,raw_img_path.split('/')[-2])
    #         os.makedirs(target_path, exist_ok=True)
    #         aligned_face_path = os.path.join(target_path, basename(raw_img_path))
            #face.save(aligned_face_path, 'JPEG')
            #print('saved ',aligned_face_path)


            # visualize_landmark(image_array=face, landmarks=landmarks)
            # plt.show()
    