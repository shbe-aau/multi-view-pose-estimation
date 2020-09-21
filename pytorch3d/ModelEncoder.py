import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder

# Network structure inspired by:
# https://arxiv.org/pdf/1708.05628.pdf (see fig. 2)
class ModelEncoder(nn.Module):
    def __init__(self, weight_file, output_size=4):
        super(ModelEncoder, self).__init__()

        self.l1 = nn.Linear(128,128)
        self.l2 = nn.Linear(128,64)
        self.l3 = nn.Linear(64,output_size)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.tanh = nn.Tanh()

        self.encoder = Encoder(weight_file)


    # Input: x = images
    # Output: y = pose as quaternion
    def forward(self,x):
        x = x.permute(0,3,1,2)
        x = self.encoder(x.float())
        x = x / torch.norm(x, dim=1).view(-1,1)

        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        y = self.l3(x)
        return y
