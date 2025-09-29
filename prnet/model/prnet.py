import torch.nn as nn
import torch


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padd=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, k_size, stride, padd)
        self.conv2 = nn.Conv2d(c_out, c_out, k_size, 1, padd)

        if stride != 1 or c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return out


class PRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # conv1
        conv1_cfg = self.config["MODEL"]["BLOCKS"]["conv1"]
        self.conv1 = nn.Conv2d(conv1_cfg[0], conv1_cfg[1], conv1_cfg[2], conv1_cfg[3], conv1_cfg[4])

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
                nn.ConvTranspose2d(block[0], block[1], block[2], block[3], block[4], output_padding=block[5])
            )
        self.decoder_blocks = nn.Sequential(*decoder_blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        print("x", x.shape)
        x = self.decoder_blocks(x)
        return x


if __name__ == "__main__":

    B, C, H, W = 2, 3, 450, 450
    x = torch.rand(B, C, H, W)
    config = {
        "MODEL": {
            "BLOCKS": {
                "conv1": [3, 16, 3, 1, 1],
                "res_blocks": [
                    [16, 32, 3, 2, 1],
                    [32, 64, 3, 2, 1],
                    [64, 128, 3, 2, 1],
                    [128, 256, 3, 2, 1],
                    [256, 512, 3, 2, 1],
                    [512, 512, 3, 2, 1],
                ],
                "decoder": [
                    [512, 256, 3, 2, 1, 1],
                    [256, 128, 3, 2, 1, 1],
                    [128, 64, 3, 2, 1, 1],
                    [64, 32, 3, 2, 1, 1],
                    [32, 16, 3, 2, 1, 1],
                    [16, 3, 3, 2, 1, 0],
                ],
            }
        }
    }
    model = PRNet(config)
    out = model(x)
    print(out.shape)
