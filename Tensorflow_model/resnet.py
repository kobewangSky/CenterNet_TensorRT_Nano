import tensorflow as tf
import numpy as np
from Tensorflow_model.utils import BasicBlock_tf, Bottleneck_tf, Deconv_tf
BN_MOMENTUM = 0.1


class PoseResNet_tf(tf.keras.Model):
    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseResNet_tf, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides = 2, padding = 'same')
        self.bn1 = tf.keras.layers.BatchNormalization(64, momentum=BN_MOMENTUM)
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPooling2D((3, 3), strides = 2, padding = 'same')

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer()
        # self.final_layer = []

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(head_conv, (3, 3), padding='same'),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(num_output, (1, 1), strides=1, padding='same')])
            else:
                fc = tf.keras.layers.Conv2D(
                    num_output,
                    (1, 1),
                    strides=1,
                    padding=0
                )
            self.__setattr__(head, fc)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.models.Sequential(
                tf.keras.layers.Conv2D(planes * block.expansion, (1, 1), strides = stride),
                tf.keras.layers.BatchNormalization(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return tf.keras.models.Sequential(layers)


    def _make_deconv_layer(self):
        layers = []
        for i in range(3):
            #layers.append(Deconv_tf(256, 2, 'same'))
            layers.append( tf.keras.layers.Conv2DTranspose( 256, (4, 4), strides = 2, padding = 'same'))
            layers.append(tf.keras.layers.BatchNormalization(256, momentum=BN_MOMENTUM))
            layers.append(tf.keras.layers.ReLU())
            self.inplanes = 256

        return tf.keras.models.Sequential(layers)

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training = training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


resnet_spec = {18: (BasicBlock_tf, [2, 2, 2, 2]),
               34: (BasicBlock_tf, [3, 4, 6, 3]),
               50: (Bottleneck_tf, [3, 4, 6, 3]),
               101: (Bottleneck_tf, [3, 4, 23, 3]),
               152: (Bottleneck_tf, [3, 8, 36, 3])}

def get_pose_net(num_layers, heads, head_conv):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet_tf(block_class, layers, heads, head_conv=head_conv)
  #model.init_weights(num_layers, pretrained=True)
  return model