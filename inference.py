import torch
import glob
import os
from detectors.ctdet import CtdetDetector
from utils.opts import opts
import cv2

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

if __name__=='__main__':
    filedir = './test_images/*.jpg'
    filelist = glob.glob(filedir)
    opt = opts().init()

    detector = CtdetDetector(opt)
    imgID = 0
    for image in filelist:
        img = cv2.imread(image)
        ret = detector.run(image)
        for classid in range(1, 80):
            result = ret['results'][classid]
            for detect in result:
                if detect[4] > 0.3:
                    img = cv2.rectangle(img, (detect[0], detect[1]), (detect[2], detect[3]), (0,255,0),3)
                    cv2.putText(img, class_name[classid], (detect[0], detect[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite('{}.jpg'.format(imgID), img)
        imgID = imgID + 1



