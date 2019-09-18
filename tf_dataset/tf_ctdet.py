import tensorflow as tf
import cv2

file_path_list = ['../data/coco/train_coco_1.record','../data/coco/train_coco_2.record','../data/coco/train_coco_3.record']
# # filename_queue = tf.train.string_input_producer(file_path_list,
# #                                                  shuffle=False,
# #                                                  num_epochs=1)
#
# reader = tf.TFRecordReader()
# key, value = reader.read(file_path_list)

raw_dataset = tf.data.TFRecordDataset(file_path_list)


image_feature_description={
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
    }

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_dataset.map(_parse_image_function)

for image_features in parsed_image_dataset:
  image_raw = image_features['image/encoded'].numpy()



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

print(1)


