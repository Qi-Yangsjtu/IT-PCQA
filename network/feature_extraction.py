import torch
import torch.nn as nn
from backbone.HSCNN import HSCNN

class Encoder(nn.Module):
    def __init__(self, resume=None, backbone='HSCNN'):
        super(Encoder, self).__init__()
        if backbone == 'HSCNN':
            self.scnn = HSCNN().cuda()
        if resume:
            self.scnn.load_state_dict(torch.load(resume),strict=False)
        self.__in_features = 256

    def forward(self, s_img1):
        dityFeat = self.scnn.forward(s_img1)
        return dityFeat

    def output_num(self):
        return self.__in_features