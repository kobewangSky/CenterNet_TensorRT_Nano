import torch
import glob
import os
from detectors.ctdet import CtdetDetector
from utils.opts import opts

if __name__=='__main__':
    filedir = './test_images/*.jpg'
    filelist = glob.glob(filedir)
    opt = opts().init()

    detector = CtdetDetector(opt)

    for image in filelist:
        ret = detector.run(image)
        print(1)