import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, len, nclass):
        super(FC, self).__init__()
        self.len = len
        self.fc = nn.Linear(self.len, nclass)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1), x
