import cv2
import numpy as np

import torch
from Pytorch_model.model import create_model
from Pytorch_model.model import load_model
from utils.image import get_affine_transform


class BaseDetector(object):
    def __init__(self, opt):
        opt.device = torch.device('cuda')
        print('Creating model...')
        self.model = create_model(opt.backbone, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True


    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)

        inp_height = (new_height | self.opt.pad) + 1
        inp_width = (new_width | self.opt.pad) + 1
        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor):
        pre_processed_images = image_or_path_or_tensor

        detections = []
        for scale in self.scales:
            images = pre_processed_images['images'][scale][0]
            meta = pre_processed_images['meta'][scale]
            meta = {k: v.numpy()[0] for k, v in meta.items()}
            images = images.to(self.opt.device)

            output, dets = self.process(images, return_time=True)
            dets = self.post_process(dets, meta, scale)
            detections.append(dets)

        results = self.merge_outputs(detections)
        return {'results': results}




