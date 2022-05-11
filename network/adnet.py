import numpy as np
import torch
import torch.nn as nn


class AdversarialNetwork(nn.Module):
  def __init__(self, hidden_size):
    super(AdversarialNetwork, self).__init__()


    self.ad_layer3 = nn.Sequential(nn.Linear(hidden_size, 64),
                                   nn.Linear(64, 1),
                                   )
    self.sigmoid = nn.Sigmoid()



  def forward(self, xfeature, yout, D_s, D_t, source_size, target_size):
    y = self.ad_layer3(xfeature)
    y = self.sigmoid(y)
    dc_target = torch.from_numpy(np.array([[1]] * source_size + [[0]] * target_size)).float().cuda()
    Dfake = torch.from_numpy(np.array([[D_s]] * source_size + [[D_t]] * target_size)).float().cuda()
    y = torch.abs(y - Dfake)
    return nn.BCELoss()(y, dc_target)