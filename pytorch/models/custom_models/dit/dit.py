import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class DiTConfig(PretrainedConfig):
    model_type = 'DiTModel'

    def __init__(
            self,
            n_blocks: int = 12,
            latent_channels: int = 4,
            context_dims: int = 768,
            n_heads: int = 16,
            head_dims: int = 128,
            model_dims: int = 768,
            intermediate_dims: int = 1024,
            scale: int = 8,
            use_linear_attn: bool = False,
            **kwargs):
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.latent_channels = latent_channels
        self.context_dims = context_dims
        self.n_heads = n_heads
        self.head_dims = head_dims
        self.model_dims = model_dims
        self.intermediate_dims = intermediate_dims
        self.scale = scale
        self.use_linear_attn = use_linear_attn


class Attention(nn.Module):
    def __init__(self, config: DiTConfig, is_cross_attn: bool = False):
        super(Attention, self).__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dims = config.head_dims

        self.channels = self.n_heads * self.head_dims
        self.to_q = nn.Linear(config.model_dims, self.channels)

        if is_cross_attn:
            self.to_k = nn.Linear(config.context_dims, self.channels)
            self.to_v = nn.Linear(config.context_dims, self.channels)
        else:
            self.to_k = nn.Linear(config.model_dims, self.channels)
            self.to_v = nn.Linear(config.model_dims, self.channels)

        self.to_o = nn.Linear(self.channels, config.model_dims)
        self.scale = self.head_dims ** -0.5

    def forward(self, q, k, v):
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        b, n_q, _ = q.shape
        b, n_k, _ = k.shape
        n_heads, head_dims = self.n_heads, self.head_dims
        q = q.reshape(b, n_q, n_heads, head_dims).permute(0, 2, 1, 3)
        k = k.reshape(b, n_k, n_heads, head_dims).permute(0, 2, 1, 3)
        v = v.reshape(b, n_k, n_heads, head_dims).permute(0, 2, 1, 3)

        if self.config.use_linear_attn:
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)
            q = q * self.scale
            scores = k.permute(0, 1, 3, 2) @ v
            output = q @ scores
        else:
            scores = q @ k.permute(0, 1, 3, 2)
            scores = scores * self.scale
            scores = torch.nn.Softmax(-1)(scores)
            output = scores @ v

        output = output.permute(0, 2, 1, 3)
        output = output.reshape(b, n_q, -1)

        output = self.to_o(output)
        return output


class DiTBlockOutput:
    def __init__(self, latent, context):
        self.latent = latent
        self.context = context


class AdaIn(nn.Module):
    def __init__(self,dims:int, config:DiTConfig):
        super(AdaIn, self).__init__()
        self.sigma = nn.Linear(config.intermediate_dims, dims)                    
        self.mu = nn.Linear(config.intermediate_dims, dims)            
        

    def forward(self, x, y):
        x = x * self.sigma(y) + self.mu(y)
        return x


class DiTBlock(nn.Module):
    def __init__(self, config: DiTConfig):
        super(DiTBlock, self).__init__()
        self.config = config
        
        self.self_attn = Attention(config=config, is_cross_attn=False)
        self.cross_attn_1 = Attention(config=config, is_cross_attn=True)
        self.cross_attn_2 = Attention(config=config, is_cross_attn=True)

        self.norm_1 = nn.LayerNorm(config.model_dims)
        self.norm_2 = nn.LayerNorm(config.model_dims)
        self.norm_3 = nn.LayerNorm(config.model_dims)

        self.adain_latent = AdaIn(config.model_dims,config=config)
        self.adain_context = AdaIn(config.context_dims,config=config)

    def forward(self, latent, timesteps, context) -> DiTBlockOutput:
        timesteps = torch.unsqueeze(timesteps,1)
        latent = self.adain_latent(latent,timesteps)
        context = self.adain_context(context,timesteps)

        latent = self.norm_1(latent)
        latent = self.self_attn(latent, latent, latent) + latent
        latent = self.norm_2(latent)
        latent = self.cross_attn_1(latent, context, context) + latent
        latent = self.norm_3(latent)
        latent = self.cross_attn_2(latent, context, context) + latent

        return DiTBlockOutput(
            latent=latent,
            context=context
        )


class DiT(nn.Module):    
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.d2s = nn.PixelUnshuffle(config.scale)
        self.s2d = nn.PixelShuffle(config.scale)

        self.timestep_block = nn.Sequential(
            nn.Linear(320, config.intermediate_dims),
            nn.GELU(),
            nn.Linear(config.intermediate_dims, config.intermediate_dims)
        )

        self.linear_entry = nn.Linear(
            config.scale * config.scale * config.latent_channels,
            config.model_dims
        )
        self.linear_exit = nn.Linear(
            config.model_dims,
            config.scale * config.scale * config.latent_channels
        )

        self.dit_blocks = nn.ModuleList(
            [
                DiTBlock(config=config)
            ] * config.n_blocks
        )

    def forward(self, latent, timesteps, context):
        latent = self.d2s(latent)
        b, c, h, w = latent.shape

        latent = torch.reshape(latent, (b, c, -1))
        latent = torch.permute(latent, (0, 2, 1))

        latent = self.linear_entry(latent)

        timesteps = self.timestep_block(timesteps)

        for layer in self.dit_blocks:
            output: DiTBlockOutput = layer(latent, timesteps, context)
            latent = output.latent

        latent = self.linear_exit(latent)

        latent = torch.permute(latent, (0, 2, 1))
        latent = torch.reshape(latent, (b, c, h, w))
        latent = self.s2d(latent)
        return latent
    

class DitModel(PreTrainedModel):
    config_class = DiTConfig
    def __init__(self, config:DiTConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = DiT(config=config)
    
    def forward(self,latent,timesteps,context):
        return self.model(latent,timesteps,context)
