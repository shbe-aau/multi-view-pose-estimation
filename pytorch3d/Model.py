import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Network structure inspired by:
# https://arxiv.org/pdf/1708.05628.pdf (see fig. 2)
class Model(nn.Module):
    def __init__(self, output_size=4):
        super(Model, self).__init__()

        self.num_views = 2
        output_size = 6 #+self.num_views
        self.l1 = nn.Linear(128,128)
        self.l2 = nn.Linear(128,64)
        self.l3 = nn.Linear(64,output_size)

        self.l4 = nn.Linear(64+output_size,self.num_views)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.tanh = nn.Tanh()


    # Input: x = lantent code
    # Output: y = pose as quaternion
    def forward(self,x):
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        #y = self.tanh(self.l3(x))
        y = self.l3(x)

        # Softmax confidences
        z = torch.cat([y, x], dim=1)
        confs = F.softmax(self.l4(z),dim=1)

        return torch.cat([confs, y], dim=1)
