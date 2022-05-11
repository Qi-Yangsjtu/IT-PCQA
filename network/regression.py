import torch.nn as nn


class Regression(nn.Module):
    def __init__(self, input):
        super(Regression, self).__init__()
        self.regression = nn.Sequential(nn.Linear(input,128),
                                        nn.Linear(128,1),)

    def forward(self, s_img1):
        out = self.regression(s_img1)
        return out
