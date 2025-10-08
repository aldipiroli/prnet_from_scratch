import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padd=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, k_size, stride, padd, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, k_size, 1, padd, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class PRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # conv1
        conv1_cfg = self.config["MODEL"]["BLOCKS"]["conv1"]
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv1_cfg[0], conv1_cfg[1], conv1_cfg[2], conv1_cfg[3], conv1_cfg[4], bias=False),
            nn.BatchNorm2d(conv1_cfg[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # residual blocks
        res_blocks_cfg = self.config["MODEL"]["BLOCKS"]["res_blocks"]
        res_blocks = []
        for block in res_blocks_cfg:
            res_blocks.append(ResBlock(block[0], block[1], block[2], block[3], block[4]))
        self.res_blocks = nn.Sequential(*res_blocks)

        # decoder
        decoder_cfg = self.config["MODEL"]["BLOCKS"]["decoder"]
        decoder_blocks = []
        for block in decoder_cfg:
            decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        block[0], block[1], block[2], block[3], block[4],
                        output_padding=block[5], bias=False
                    ),
                    nn.BatchNorm2d(block[1]),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )
        self.decoder_blocks = nn.Sequential(*decoder_blocks)

        # output layer
        out_cfg = self.config["MODEL"]["BLOCKS"]["out_layer"]
        self.out_layer = nn.Conv2d(out_cfg[0], out_cfg[1], out_cfg[2], out_cfg[3], out_cfg[4])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.decoder_blocks(x)
        coords = self.out_layer(x)
        return coords
