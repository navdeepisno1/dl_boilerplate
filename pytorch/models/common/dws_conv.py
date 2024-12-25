import torch
import torch.nn as nn


class DepthWiseSeparableConv2D(nn.Module):
    """Some Information about DepthWiseSeparableConv2D"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 expansion_factor: int = 6,
                 stride: int = 1
                 ):
        super(DepthWiseSeparableConv2D, self).__init__()
        self.dws = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels*expansion_factor,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels
        )
        self.conv_1x1 = nn.Conv2d(
            in_channels=in_channels*expansion_factor,
            out_channels=out_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x):
        x = self.dws(x)
        x = self.conv_1x1(x)
        return x
