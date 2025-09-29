import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
from prnet.model.prnet import PRNet
from prnet.utils.misc import load_config


def test_model():
    config = load_config("prnet/config/prnet_config.yaml")
    model = PRNet(config)
    B, C, H, W = 2, 3, 512, 512
    x = torch.randn(B, C, H, W)
    out = model(x)
    assert out.shape == (B, 3, H, W)

if __name__ == "__main__":
    print("All tests passed!")
