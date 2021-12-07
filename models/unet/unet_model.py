"""
Full assembly of the parts to form the complete network 

Adapted from https://github.com/milesial/Pytorch-UNet
"""

from torch import nn

from .unet_parts import DoubleConv, Down, OutConv, ConvThenUp
from config_loader import config


class UNet(nn.Module):
    """ In the ConvMOS architecture this is supposed to be used as a type of Global Module """
    def __init__(self, n_classes: int = 1, bilinear: bool = True):
        super(UNet, self).__init__()

        # This uses all aux variables, the temperature/precipitation (+1), and elevation (+1)
        input_depth = (
            len(
                list(
                    filter(None, config.get('DataOptions', 'aux_variables').split(','))
                )
            )
            + 2
        )
        self.n_channels = input_depth
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        # factor = 1
        self.inc = DoubleConv(input_depth, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        # self.down2 = Down(64, 128 // factor)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = ConvThenUp(128, 64 // factor, mid_channels=128, bilinear=bilinear)
        self.up3 = ConvThenUp(128, 64, mid_channels=128, bilinear=bilinear)
        self.up4 = ConvThenUp(64 * factor, 32, mid_channels=64, bilinear=bilinear)
        self.outdc = DoubleConv(64, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        x = self.outdc(x)
        out = self.outc(x)
        return out
