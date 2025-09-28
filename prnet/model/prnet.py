import torch
import torch.nn as nn

class PRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(1,1)

    def forward(self, img):
        return 

