import torch
import torch.nn as nn


class Activation(nn.Module):
    def __init__(self, activation: str = 'relu'):
        super(Activation, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(x)
        return x
