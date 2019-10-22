from pycocotools.coco import COCO
from random import shuffle
import tensorflow as tf
import os
import cv2
import numpy as np
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import flip, color_aug
import math

def decode_img(img, IMG_WIDTH, IMG_HEIGHT):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

class TF_dataloader(object):
    num_classes = 80
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, imgs_dir, annotations_filepath, batch_size, opt , shuffle_img = True ):
        self.current_batch = 0
        self.imgs_dir = imgs_dir
        self.annotations_filepath = annotations_filepath
        self.shuffle_img = shuffle_img
        self.batch_size = batch_size

        self.coco = COCO(self.annotations_filepath)
        self.img_ids = self.coco.getImgIds()  # totally 82783 images
        self.cat_ids = self.coco.getCatIds()
        if self.shuffle_img:
            shuffle(self.img_ids)
        self.opt = opt

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __len__(self):
        return self.num_samples


    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i


    def get_coco_data(self, split):
        coco_data = []
        for index in range(self.current_batch*self.batch_size, (self.current_batch + 1) * self.batch_size):
            img_id = self.img_ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ids=ann_ids)
            num_objs = self.batch_size
            self.max_objs = self.batch_size

            img_detail = self.coco.loadImgs(img_id)[0]

            path = os.path.join(self.imgs_dir, img_detail['file_name'])
            img = cv2.imread(path)

            height, width = img.shape[0], img.shape[1]
            c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

            flipped = False
            if split == 'train':
                if not self.opt.not_rand_crop:
                    s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                    w_border = self._get_border(128, img.shape[1])
                    h_border = self._get_border(128, img.shape[0])
                    c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                    c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
                else:
                    sf = self.opt.scale
                    cf = self.opt.shift
                    c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                    c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                    s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

                if np.random.random() < self.opt.flip:
                    flipped = True
                    img = img[:, ::-1, :]
                    c[0] = width - c[0] - 1

            trans_input = get_affine_transform(
                c, s, 0, [input_w, input_h])
            inp = cv2.warpAffine(img, trans_input,
                                 (input_w, input_h),
                                 flags=cv2.INTER_LINEAR)
            inp = (inp.astype(np.float32) / 255.)
            # if split == 'train' and not self.opt.no_color_aug:
            #     color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
            inp = (inp - self.mean) / self.std
            inp = inp.transpose(2, 0, 1)

            output_h = input_h // self.opt.down_ratio
            output_w = input_w // self.opt.down_ratio
            num_classes = self.num_classes
            trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

            hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
            dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)
            ind = np.zeros((self.max_objs), dtype=np.int64)
            reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
            cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
            cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

            draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                draw_umich_gaussian

            gt_det = []
            for k in range(num_objs):
                ann = anns[k]
                bbox = self._coco_box_to_bbox(ann['bbox'])
                cls_id = int(self.cat_ids[ann['category_id']])
                if flipped:
                    bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_gaussian(hm[cls_id], ct_int, radius)
                    wh[k] = 1. * w, 1. * h
                    ind[k] = ct_int[1] * output_w + ct_int[0]
                    reg[k] = ct - ct_int
                    reg_mask[k] = 1
                    cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                    cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                    gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                                   ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

            ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}

            coco_data.append(ret)
        self.current_batch = self.current_batch + 1
        return coco_data