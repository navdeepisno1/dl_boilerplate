import torch
import torch.nn as nn

class UpSample2D(nn.Module):
    """Some Information about UpSample2D"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 kernel_size: int = 3,
                 padding: int = 1,                                  
                 scale_factor:int = 2                 
                 ):
        super(UpSample2D, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,            
            padding=padding
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    def forward(self, x):
        x = self.conv(x)    
        x = self.upsample(x)
        return x
    
class DownSample2D(nn.Module):
    """Some Information about UpSample2D"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 kernel_size: int = 3,
                 padding: int = 1,                                  
                 stride:int = 2                 
                 ):
        super(UpSample2D, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,            
            padding=padding,
            stride=stride
        )        
    def forward(self, x):
        x = self.conv(x)            
        return x