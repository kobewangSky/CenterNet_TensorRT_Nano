import tensorflow as tf
import cv2
import numpy as np

# # filename_queue = tf.train.string_input_producer(file_path_list,
# #                                                  shuffle=False,
# #                                                  num_epochs=1)
#
# reader = tf.TFRecordReader()
# key, value = reader.read(file_path_list)

class TF_ctdet(object):
    num_classes = 80
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    def GetTFRecordData(file_path_list, batch_size ):

        raw_dataset = tf.data.TFRecordDataset(file_path_list)

        def _parse_image_function(example_proto):
          # Parse the input tf.Example proto using the dictionary above.
          image_feature_description = {
              'image/height': tf.io.FixedLenFeature([], tf.int64),
              'image/width': tf.io.FixedLenFeature([], tf.int64),
              'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
              'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
              'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
              'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
              'image/object/class/label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
              'image/encoded': tf.io.FixedLenFeature([], tf.string),
              'image/format': tf.io.FixedLenFeature([], tf.string),
          }
          example = tf.io.parse_single_example(example_proto, image_feature_description)
          hight = tf.cast(example['image/height'], tf.int64)
          width = tf.cast(example['image/width'], tf.int64)
          xmin = tf.cast(example['image/object/bbox/xmin'], tf.float32 )
          xmax = tf.cast(example['image/object/bbox/xmax'], tf.float32 )
          ymin = tf.cast(example['image/object/bbox/ymin'], tf.float32 )
          ymax = tf.cast(example['image/object/bbox/ymax'], tf.float32 )
          label = tf.cast(example['image/object/class/label'], tf.int64 )
          image = tf.io.decode_raw(example['image/encoded'], tf.uint8)
          return xmin, xmax, ymin, ymax, label, image

        #train_dataset = raw_dataset.batch(10)
        # train_dataset = train_dataset.prefetch(
        #     buffer_size=tf.data.experimental.AUTOTUNE)

        raw_dataset = raw_dataset.map(_parse_image_function)
        raw_dataset = raw_dataset.padded_batch( batch_size, padded_shapes=([None],[None],[None],[None],[None],[None]))

        iterator  = tf.compat.v1.data.make_one_shot_iterator(raw_dataset)

        item = iterator.get_next()

        return item





# for image_features in parsed_image_dataset:
#     height = image_features['image/height']
#     width = image_features['image/width']
#     xmin = image_features['image/object/bbox/xmin']
#     xmax = image_features['image/object/bbox/xmax']
#     ymin = image_features['image/object/bbox/ymin']
#     ymax = image_features['image/object/bbox/ymax']
#     label = image_features['image/object/class/label']
#
#     image = image_features['image/encoded']
#     format = image_features['image/format']
#     print(1)
# hight = tf.cast(features['image/height'], tf.int64 )
# width = tf.cast(features['image/width'], tf.int64 )
# xmin = tf.cast(features['image/object/bbox/xmin'], tf.float )
# xmax = tf.cast(features['image/object/bbox/xmax'], tf.float )
# ymin = tf.cast(features['image/object/bbox/ymin'], tf.float )
# ymax = tf.cast(features['image/object/bbox/ymax'], tf.float )
# label = tf.cast(features['image/object/class/label'], tf.int64 )
#
# image = tf.decode_raw(features['image/encoded'], tf.uint8 )
# format = tf.decode_raw(features['image/format'], tf.uint8 )

if __name__ == "__main__":
    file_path_list = ['../data/coco/train_coco_1.record', '../data/coco/train_coco_2.record',
                      '../data/coco/train_coco_3.record']
    TF_ctdet = TF_ctdet()
    Dataitem = TF_ctdet.GetTFRecordData(file_path_list, 6)
    print(1)

