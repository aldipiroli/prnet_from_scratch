import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from moco.utils.augmentations import MoCoAugmentations
from moco.utils.misc import load_config


def test_augmentations():
    config = load_config("moco/config/moco_config.yaml")
    moco_augm = MoCoAugmentations(config=config)
    B, C, H, W = 2, 3, 512, 512
    x = torch.randn(B, C, H, W)
    x_t = moco_augm.augment(x)
    crop_size = config["DATA"]["augm"]["crop_size"]
    assert x_t.shape == (B, 3, crop_size[0], crop_size[1])


if __name__ == "__main__":
    print("All tests passed!")
