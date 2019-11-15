import torch
import glob
import os
from detectors.ctdet import CtdetDetector
from utils.opts import opts
import cv2
import time
import numpy as np

class_name = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

if __name__=='__main__':
    opt = opts().init()
    detector = CtdetDetector(opt)
    imgID = 0
    if opt.demo == 'Rpicam':
        print(gstreamer_pipeline(flip_method=0))
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            # Window
            while 1:
                ret_val, img = cap.read()
                cv2.imwrite('{}.jpg'.format(imgID), img)
                imgID = imgID + 1

                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 27:
                    break
            cap.release()
        else:
            print("Unable to open camera")
    elif opt.demo == 'Webcam':
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Window
            while 1:
                ret_val, img = cap.read()
                start = time.time()
                ret = detector.run(img)
                end = time.time()
                print(end - start)
                for classid in range(1, 80):
                    result = ret['results'][classid]
                    for detect in result:
                        if detect[4] > 0.2:
                            img = cv2.rectangle(img, (detect[0], detect[1]), (detect[2], detect[3]), (0, 255, 0), 3)
                            cv2.putText(img, class_name[classid], (detect[0], detect[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.imwrite('{}.jpg'.format(imgID), img)
                imgID = imgID + 1

                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 27:
                    break
            cap.release()
        else:
            print("Unable to open camera")

    else:

        filedir = './test_images/*.jpg'
        filelist = glob.glob(filedir)

        inferencetime = 0
        TimeList = []

        while (len(TimeList) < 100):
            for file in filelist:
                img = cv2.imread(file)
                start = time.time()
                ret = detector.run(img)
                end = time.time()
                print(end - start)
                TimeList.append(end - start)
                for classid in range(1, 80):
                    result = ret['results'][classid]
                    for detect in result:
                        if detect[4] > 0.7:
                            img = cv2.rectangle(img, (detect[0], detect[1]), (detect[2], detect[3]), (0,255,0),3)
                            cv2.putText(img, class_name[classid], (detect[0], detect[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

                cv2.imwrite('{}.jpg'.format(imgID), img)
                imgID = imgID + 1
        temp = np.array(TimeList[1:])
        print('mean = {}'.format(temp.mean()))



#ctdet --exp_id coco_res18 --backbone res_18 --batch_size 1 --load_model ./exp/ctdet/coco_res18/model_best.pth --demo Rpicam