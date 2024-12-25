import torch
import torch.nn as nn
from typing import List, Union, Optional, Any
from .activations import Activation


class ResidualBlock2D(nn.Module):
    """Some Information about ResidualBlock2D"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            n_intermediate_layers: int = 1,
            activation: str = 'relu'
    ):
        super(ResidualBlock2D, self).__init__()
        self.entry_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding
            ),
            Activation(activation=activation),
            nn.BatchNorm2d(out_channels)
        )

        self.final_block = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                )
            ] * n_intermediate_layers
        )

        self.final_act = Activation(activation=activation)

        if in_channels == out_channels:
            self.skip_block = nn.Identity()
        else:
            self.skip_block = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0
            )

    def forward(self, x):
        residual = x
        x = self.entry_block(x)

        for i, layer in enumerate(self.final_block):
            x = layer(x)

        x = self.final_act(x)

        x = x + self.skip_block(residual)
        return x
    

class ResidualBlock1D(nn.Module):
    """Some Information about ResidualBlock1D"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_intermediate_layers: int = 1,
            activation: str = 'relu'
    ):
        super(ResidualBlock1D, self).__init__()
        self.entry_block = nn.Sequential(
            nn.Linear(
                in_features=in_channels,
                out_features=out_channels                
            ),
            Activation(activation=activation),
            nn.LayerNorm(out_channels)
        )

        self.final_block = nn.ModuleList(
            [
                nn.Linear(
                in_features=out_channels,
                out_features=out_channels                
            ),
            ] * n_intermediate_layers
        )

        self.final_act = Activation(activation=activation)

        if in_channels == out_channels:
            self.skip_block = nn.Identity()
        else:
            self.skip_block = nn.Linear(
                in_features=in_channels,
                out_features=out_channels                
            ),

    def forward(self, x):
        residual = x
        x = self.entry_block(x)

        for i, layer in enumerate(self.final_block):
            x = layer(x)

        x = self.final_act(x)

        x = x + self.skip_block(residual)
        return x
