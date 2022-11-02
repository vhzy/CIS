import sys
import cv2
import dlib
import os
import glob

def video2image(video_path):
    cap = cv2.VideoCapture(video_path)
    prefix =  video_path.split('/')[-1]
    prefix = prefix[9:14]
    ids = 0
    while True:
        ret, img = cap.read()
        if img is None:
            break
        ids += 1
        target_path = f'/home/hfutzny/sda/casual_face/CIS/data/DISFA/{prefix}'
        os.makedirs(target_path,exist_ok=True)
        cv2.imwrite(os.path.join(target_path,str(ids)+'.jpg'),img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
raw_path = '/home/hfutzny/sda/casual_face/dataset/Videos_LeftCamera'
dirs = glob.glob(os.path.join(raw_path, '*.avi'))
# print(dirs)
for dir in dirs:
    video2image(dir)
 
# if len(sys.argv) > 2 or "-h" in sys.argv or "--help" in sys.argv:
#     _help()
# elif len(sys.argv) == 2:
#     face_detect(sys.argv[1])
# else:
#     face_detect()