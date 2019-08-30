import torch
import numpy as np
import torch.nn as nn
from Pytorch_model.DCN.utils import ConvOffset2D
import torch.nn.functional as F

class DeformConvNet(nn.Module):
    def __init__(self, inplanes ,planes, kernel, stride, padding):
        super(DeformConvNet, self).__init__()

        # conv
        self.offset12 = ConvOffset2D(inplanes)
        self.conv12 = nn.Conv2d(inplanes, planes, kernel, padding=padding, stride=stride)
        self.bn12 = nn.BatchNorm2d(planes)

    def forward(self, x):

        x = self.offset12(x)
        x = F.relu(self.conv12(x))
        x = self.bn12(x)

        return x

    def freeze(self, module_classes):
        '''
        freeze modules for finetuning
        '''
        for k, m in self._modules.items():
            if any([type(m) == mc for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = False

    def unfreeze(self, module_classes):
        '''
        unfreeze modules
        '''
        for k, m in self._modules.items():
            if any([isinstance(m, mc) for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = True

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DeformConvNet, self).parameters())