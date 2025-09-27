import torch.nn as nn


class MoCoLoss(nn.Module):
    def __init__(self, config, logger):
        super(MoCoLoss, self).__init__()
        self.config = config
        self.logger = logger

    def forward(sefl, logits, labels):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        loss_dict = {"contrastive_loss": loss}
        return loss, loss_dict
