import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Network structure inspired by:
# https://arxiv.org/pdf/1708.05628.pdf (see fig. 2)
class Model(nn.Module):
    def __init__(self, num_views=4):
        super(Model, self).__init__()

        self.num_views = num_views
        self.output_size = self.num_views*(6+1)

        # Upscale lantent vector
        self.l01 = nn.Linear(128,128)
        self.l02 = nn.Linear(128,128)
        self.l03 = nn.Linear(128,128)
        self.l04 = nn.Linear(128,128)

        self.bn01 = nn.BatchNorm1d(128)
        self.bn02 = nn.BatchNorm1d(128)
        self.bn03 = nn.BatchNorm1d(128)
        self.bn04 = nn.BatchNorm1d(128)

        # Regress pose
        self.l1 = nn.Linear(128,128)
        self.l2 = nn.Linear(128,64)
        self.l3 = nn.Linear(64,self.output_size)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)


    # Input: x = lantent code
    # Output: y = pose as quaternion
    def forward(self,x):
        x = F.relu(self.bn01(self.l01(x)))
        x = F.relu(self.bn02(self.l02(x)))
        x = F.relu(self.bn03(self.l03(x)))
        x = F.relu(self.bn04(self.l04(x)))

        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        y = self.l3(x)
        confs = F.softmax(y[:,:self.num_views], dim=1)
        return torch.cat([confs, y[:,self.num_views:]], dim=1)