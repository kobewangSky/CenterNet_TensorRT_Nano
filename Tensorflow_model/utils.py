import tensorflow as tf
import numpy as np

layers = tf.keras.layers
BN_MOMENTUM = 0.1

def conv3x3_tf(in_planes, out_planes, stride = 1):
    return layers.Conv2D(out_planes, (3, 3), strides = stride)

class BasicBlock_tf(tf.keras.Model):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride, downsample = None):
        super(BasicBlock_tf, self).__init__()
        self.conv1 = conv3x3_tf(in_planes, out_planes, stride)
        self.bn1 = layers.BatchNormalization(out_planes, momentum=BN_MOMENTUM)
        self.relu = layers.ReLU(inplace=True)
        self.conv2 = conv3x3_tf(out_planes, out_planes)
        self.bn2 = layers.BatchNormalization(out_planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def call(self, input_tensor, training = False):
        out = self.conv1(input_tensor)
        out = self.bn1(out, training = training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training)

        if self.downsample is not None:
            residual = self.downsample(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_tf(tf.keras.Model):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_tf, self).__init__()
        self.conv1 = layers.Conv2D(planes, (1, 1), strides = stride)

        self.bn1 = layers.BatchNormalization(planes, momentum=BN_MOMENTUM)
        self.conv2 = layers.Conv2D(planes, (3, 3), strides = stride)

        self.bn2 = layers.BatchNormalization(planes, momentum=BN_MOMENTUM)

        self.conv3 = layers.Conv2D(planes * self.expansion, (1, 1), strides = stride)

        self.bn3 = layers.BatchNormalization(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = layers.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def call(self, input_tensor ,training = False):
        residual = input_tensor

        out = self.conv1(input_tensor)
        out = self.bn1(out, training = training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training = training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training = training)

        if self.downsample is not None:
            residual = self.downsample(input_tensor)

        out += residual
        out = self.relu(out)

        return out
