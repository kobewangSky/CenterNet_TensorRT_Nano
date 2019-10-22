import numpy as np
import tensorflow as tf
from TF_train.losses import _neg_loss

def _sigmoid(x):
  #y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  y = tf.clip_by_value(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class CtdetLoss(object):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.opt = opt

    def forward(self, batch , outputs):
        loss = _neg_loss(batch, outputs)
        return loss

# class CtdetTrainer(BaseTrainer):
#     def __init__(self, opt, model, optimizer=None):
#         super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
#         self.model = model
#
#     def _get_losses(self, opt):
#         loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
#         loss = CtdetLoss(opt)
#         return loss_states, loss
#
#     def debug(self, batch, output, iter_id):
#         assert False, 'CtdetTrainer debug'
#
#     def save_result(self, output, batch, results):
#         assert False, 'CtdetTrainer save_result'
