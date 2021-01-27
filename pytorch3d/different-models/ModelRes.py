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

        # Block 1
        self.l11 = nn.Linear(128,128)
        self.l12 = nn.Linear(128,128)

        self.bn11 = nn.BatchNorm1d(128)
        self.bn12 = nn.BatchNorm1d(128)

        # Block 2
        self.l21 = nn.Linear(128,128)
        self.l22 = nn.Linear(128,128)

        self.bn21 = nn.BatchNorm1d(128)
        self.bn22 = nn.BatchNorm1d(128)

        # Block 3
        self.l31 = nn.Linear(128,128)
        self.l32 = nn.Linear(128,128)

        self.bn31 = nn.BatchNorm1d(128)
        self.bn32 = nn.BatchNorm1d(128)

        # Block 4
        self.l41 = nn.Linear(128,128)
        self.l42 = nn.Linear(128,128)

        self.bn41 = nn.BatchNorm1d(128)
        self.bn42 = nn.BatchNorm1d(128)

        # Regress pose using the feature vector
        self.fc1 = nn.Linear(128,self.output_size)


    # Input: x = lantent code
    # Output: y = pose as quaternion
    def forward(self,x0):
        # Block 1
        x1 = F.relu(self.bn11(self.l11(x0)))
        x1 = F.relu(self.bn12(self.l12(x1)))

        # Block 2
        x2 = F.relu(self.bn21(self.l21(x0 + x1)))
        x2 = F.relu(self.bn22(self.l22(x2)))

        # Block 3
        x3 = F.relu(self.bn31(self.l31(x1 + x2)))
        x3 = F.relu(self.bn32(self.l32(x3)))

        # Block 4
        x4 = F.relu(self.bn41(self.l41(x2 + x3)))
        x4 = F.relu(self.bn42(self.l42(x4)))

        y = self.fc1(x4)
        confs = F.softmax(y[:,:self.num_views], dim=1)
        return torch.cat([confs, y[:,self.num_views:]], dim=1)
