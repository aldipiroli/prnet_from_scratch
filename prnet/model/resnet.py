import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=False, normalize_output=True):
        super().__init__()
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        resnet = models.resnet18(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.projection = nn.Linear(512, 128)
        self.normalize_output = normalize_output

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Avg. pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Project to lower dim
        x = self.projection(x)
        if self.normalize_output:
            x = nn.functional.normalize(x, dim=1)

        return x


class ResNet18Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = ResNet18Backbone()
        if self.config["MODEL"]["disable_bn"]:
            self.backbone = replace_batchnorm_with_identity(self.backbone)

    def forward(self, img):
        out = self.backbone(img)
        return out


def replace_batchnorm_with_identity(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr_name, nn.Identity())
    return model


if __name__ == "__main__":
    model = ResNet18Model()
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(y.shape)  # Expected: [1, 128, 32, 32]
