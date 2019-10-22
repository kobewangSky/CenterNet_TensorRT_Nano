import tensorflow as tf
import numpy as np

def _neg_loss(preds, gt):
    _, h, w, c = preds.get_shape().as_list()
    num_pos = tf.reduce_sum(tf.cast(gt == 1, tf.float32))

    neg_weights = tf.pow(1 - gt, 4)
    pos_weights = tf.ones_like(preds, dtype=tf.float32)
    weights = tf.where(gt == 1, pos_weights, neg_weights)
    inverse_preds = tf.where(gt == 1, preds, 1 - preds)

    loss = tf.log(inverse_preds + 0.0001) * tf.pow(1 - inverse_preds, 2) * weights
    loss = tf.reduce_mean(loss)
    loss = -loss / (num_pos + 1)
    return loss