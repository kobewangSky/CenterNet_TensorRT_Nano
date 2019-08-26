import torch
import numpy as np
from trains.losses import RegL1Loss


def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = torch.nn.L1Loss(reduction='sum')
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
        if not opt.mse_loss:
            output['hm'] = _sigmoid(output['hm'])
        hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
        if opt.wh_weight > 0:
            wh_loss = self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh']) / opt.num_stacks
