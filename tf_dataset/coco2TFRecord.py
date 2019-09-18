from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os, sys
import numpy as np
import tensorflow as tf
import logging

import tf_dataset.utils as utils
import argparse


def load_coco_dection_dataset(imgs_dir, annotations_filepath, shuffle_img = True ):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of coco images
        annotations_filepath: file path of coco annotations file
        shuffle_img: wheter to shuffle images order
    Return:
        coco_data: list of dictionary format information of each image
    """
    coco = COCO(annotations_filepath)
    img_ids = coco.getImgIds() # totally 82783 images
    cat_ids = coco.getCatIds() # totally 90 catagories, however, the number of categories is not continuous, \
                               # [0,12,26,29,30,45,66,68,69,71,83] are missing, this is the problem of coco dataset.

    if shuffle_img:
        shuffle(img_ids)

    coco_data = []

    nb_imgs = len(img_ids)
    for index, img_id in enumerate(img_ids):
        if index % 100 == 0:
            print("Readling images: %d / %d "%(index, nb_imgs))
        img_info = {}
        bboxes = []
        labels = []

        img_detail = coco.loadImgs(img_id)[0]
        pic_height = img_detail['height']
        pic_width = img_detail['width']

        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            bboxes_data = ann['bbox']
            bboxes_data = [bboxes_data[0]/float(pic_width), bboxes_data[1]/float(pic_height),\
                                  bboxes_data[2]/float(pic_width), bboxes_data[3]/float(pic_height)]
                         # the format of coco bounding boxs is [Xmin, Ymin, width, height]
            bboxes.append(bboxes_data)
            labels.append(ann['category_id'])


        img_path = os.path.join(imgs_dir, img_detail['file_name'])
        img_bytes = tf.io.gfile.GFile(img_path,'rb').read()

        img_info['pixel_data'] = img_bytes
        img_info['height'] = pic_height
        img_info['width'] = pic_width
        img_info['bboxes'] = bboxes
        img_info['labels'] = labels

        coco_data.append(img_info)
    return coco_data


def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': utils.int64_feature(img_data['height']),
        'image/width': utils.int64_feature(img_data['width']),
        'image/object/bbox/xmin': utils.float_list_feature(xmin),
        'image/object/bbox/xmax': utils.float_list_feature(xmax),
        'image/object/bbox/ymin': utils.float_list_feature(ymin),
        'image/object/bbox/ymax': utils.float_list_feature(ymax),
        'image/object/class/label': utils.int64_list_feature(img_data['labels']),
        'image/encoded': utils.bytes_feature(img_data['pixel_data']),
        'image/format': utils.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example

def main(opt):
    if opt.set == "train":
        imgs_dir = os.path.join(opt.data_dir,'images', 'train2017')
        annotations_filepath = os.path.join(opt.data_dir,'annotations','instances_train2017.json')
        print("Convert coco train file to tf record")
    elif opt.set == "val":
        imgs_dir = os.path.join(opt.data_dir,'images', 'val2017')
        annotations_filepath = os.path.join(opt.data_dir,'annotations','instances_val2017.json')
        print("Convert coco val file to tf record")
    else:
        raise ValueError("you must either convert train data or val data")
    # load total coco data
    coco_data = load_coco_dection_dataset(imgs_dir,annotations_filepath,shuffle_img=opt.shuffle_imgs)
    total_imgs = len(coco_data)

    #total_imgs = 0


    # write coco data to tf record

    # with tf.python_io.TFRecordWriter(opt.output_filepath) as tfrecord_writer:
    #     for index, img_data in enumerate(coco_data):
    #         if index % 100 == 0:
    #             print("Converting images: %d / %d" % (index, total_imgs))
    #         example = dict_to_coco_example(img_data)
    #         tfrecord_writer.write(example.SerializeToString())

    Savefile = 1
    start = 0
    end = 0
    while Savefile <= int(opt.batch_size):

        FileName_list = os.path.basename(opt.output_filepath).split('.')
        Savefilepath_temp = os.path.join(os.path.dirname(opt.output_filepath), FileName_list[0]) + '_{}.'.format(Savefile) + FileName_list[1]


        tfrecord_writer = tf.python_io.TFRecordWriter(Savefilepath_temp)

        start = end
        if Savefile == opt.batch_size:
            end = total_imgs
        else:
            end = int((total_imgs * Savefile) / opt.batch_size)


        for index in range(start, end):
            if index % 100 == 0:
                print("Converting images: %d / %d" % (index, total_imgs))
            example = dict_to_coco_example(coco_data[index])
            tfrecord_writer.write(example.SerializeToString())

        Savefile = Savefile + 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", help="coco path")
    parser.add_argument("-s", "--set", help="train or val", default="train")
    parser.add_argument("-o", "--output_filepath", help="Path to output TFRecord")
    parser.add_argument("-k", "--shuffle_imgs", help="shuffle_imgs", action='store_true')
    parser.add_argument("-b", "--batch_size", default= 1,help="set batch size for one save tfrecord")
    args = parser.parse_args()
    main(args)