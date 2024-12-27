import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class BBoxAEModelConfig(PretrainedConfig):
    model_type = 'BBoxAEModel'

    def __init__(self,bbox_type:str = 'xyxy', **kwargs):
        super().__init__(**kwargs)
        self.n_points = len(bbox_type)
        
            

class ResidualBlock1D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_intermediate_layers: int = 1,
    ):
        super(ResidualBlock1D, self).__init__()
        self.entry_block = nn.Sequential(
            nn.Linear(
                in_features=in_channels,
                out_features=out_channels
            ),
            nn.GELU(),
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

        self.final_act = nn.GELU()

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


class BBoxEncoderModel(nn.Module):
    def __init__(self, config: BBoxAEModelConfig):
        super(BBoxEncoderModel, self).__init__()
        self.entry_block = ResidualBlock1D()
        self.encoder_blocks = [
            ResidualBlock1D()
        ]

    def forward(self, x):

        return x


class BBoxDecoderModel(nn.Module):
    def __init__(self, config: BBoxAEModelConfig):
        super(BBoxDecoderModel, self).__init__()

    def forward(self, x):

        return x


class BBoxAEModel(PreTrainedModel):
    config_class = BBoxAEModelConfig

    def __init__(self, config: BBoxAEModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.encoder = BBoxEncoderModel(config=config)
        self.decoder = BBoxDecoderModel(config=config)

    def forward(self, x):
        e_x = self.encoder(x)
        y = self.decoder(e_x)
        return y
