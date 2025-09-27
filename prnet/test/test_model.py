import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from moco.model.resnet import ResNet18Model, replace_batchnorm_with_identity
from moco.utils.misc import load_config


def test_model():
    config = load_config("moco/config/moco_config.yaml")
    model = ResNet18Model(config)
    B, C, H, W = 2, 3, 512, 512
    x = torch.randn(B, C, H, W)
    out = model(x)
    assert out.shape == (B, 128)


def test_replace_batchnorm_with_identity():
    config = load_config("moco/config/moco_config.yaml")
    config["MODEL"]["disable_bn"] = False
    model = ResNet18Model(config)
    model_no_bn = replace_batchnorm_with_identity(model)
    assert isinstance(model_no_bn.backbone.bn1, nn.Identity)


if __name__ == "__main__":
    print("All tests passed!")
