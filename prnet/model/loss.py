import torch.nn as nn


class FaceLoss(nn.Module):
    def __init__(self, config, logger):
        super(FaceLoss, self).__init__()
        self.config = config
        self.logger = logger

    def forward(sefl, uv_coords, gt_uv_coords):
        B, C, H, W = uv_coords.shape

        loss_fn = nn.MSELoss()
        loss_uv_coords = loss_fn(uv_coords, gt_uv_coords)

        loss = loss_uv_coords 
        loss_dict = {}
        loss_dict["loss_uv_coords"] = loss_uv_coords
        return loss, loss_dict
