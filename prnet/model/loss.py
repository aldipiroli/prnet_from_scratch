import torch.nn as nn


class FaceLoss(nn.Module):
    def __init__(self, config, logger):
        super(FaceLoss, self).__init__()
        self.config = config
        self.logger = logger

    def forward(sefl, preds, labels):
        loss_fn = nn.MSELoss()
        loss = loss_fn(preds, labels)
        loss_dict = {}
        loss_dict["mse"] = loss
        return loss, loss_dict
