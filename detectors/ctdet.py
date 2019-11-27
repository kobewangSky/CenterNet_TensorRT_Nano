import numpy as np

import torch
from detectors.base_detector import BaseDetector
from Pytorch_model.utils import ctdet_decode
from Pytorch_model.utils import ctdet_post_process
import time
import subprocess
import os
class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model_trt(images)
            output = self.model(output)[-1]

            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = ctdet_decode(hm, wh, reg=reg, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def UseTensorrt(self):
        onnx_path = "temp.onnx"
        image_shape = torch.ones((1, 3, 512, 512)).cuda()
        torch.onnx.export(
            self.model,
            image_shape,
            onnx_path,
            verbose=False,
            input_names=['image'],
        )
        tensorrt_name = onnx_path.replace('onnx', 'trt')
        cmd = 'onnx2trt {} -o {}'.format(onnx_path, tensorrt_name)
        subprocess.call(cmd, shell=True)