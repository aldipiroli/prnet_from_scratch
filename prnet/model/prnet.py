import torch.nn as nn
import torch
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padd=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, k_size, stride, padd)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, k_size, 1, padd)

        if stride != 1 or c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)
        return out


class PRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # conv1
        conv1_cfg = self.config["MODEL"]["BLOCKS"]["conv1"]
        self.conv1 = nn.Conv2d(conv1_cfg[0], conv1_cfg[1], conv1_cfg[2], conv1_cfg[3], conv1_cfg[4])
        self.relu = nn.ReLU(inplace=True)

        # res_blocks
        res_blocks_cfg = self.config["MODEL"]["BLOCKS"]["res_blocks"]
        res_blocks = []
        for i in range(len(res_blocks_cfg)):
            block = res_blocks_cfg[i]
            res_blocks.append(ResBlock(block[0], block[1], block[2], block[3], block[4]))
        self.res_blocks = nn.Sequential(*res_blocks)

        # decoder
        decoder_cfg = self.config["MODEL"]["BLOCKS"]["decoder"]
        decoder_blocks = []
        for i in range(len(decoder_cfg)):
            block = decoder_cfg[i]
            decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(block[0], block[1], block[2], block[3], block[4], output_padding=block[5]),
                    nn.ReLU(inplace=True),
                )
            )
        self.decoder_blocks = nn.Sequential(*decoder_blocks)

        # uv_coords
        out_cfg = self.config["MODEL"]["BLOCKS"]["out_layer"]
        self.out_layer = nn.Conv2d(out_cfg[0], out_cfg[1], out_cfg[2], out_cfg[3], out_cfg[4])

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        x = self.decoder_blocks(x)
        x = self.out_layer(x)
        return x
