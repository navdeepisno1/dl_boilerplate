import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class VitConfig(PretrainedConfig):
    def __init__(
            self,
            n_heads: int = 16,
            head_dims: int = 64,
            model_dims: int = 768,
            input_channels: int = 3,
            patch_size: int = 14,
            intermediate_dims: int = 1024,
            image_size: int = 448,
            use_linear_attn: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.head_dims = head_dims
        self.model_dims = model_dims
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.intermediate_dims = intermediate_dims
        self.image_size = image_size
        self.use_linear_attn = use_linear_attn


class ViTOutput:
    def __init__(
            self,
            last_hidden_state,
            hidden_states,
            attention_scores
    ):
        self.last_hidden_state = last_hidden_state,
        self.hidden_states = hidden_states,
        self.attention_scores = attention_scores


class Attention(nn.Module):
    def __init__(self, config: VitConfig, is_cross_attn: bool = False):
        super(Attention, self).__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dims = config.head_dims

        self.channels = self.n_heads * self.head_dims
        self.to_q = nn.Linear(config.model_dims, self.channels)

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
        return output, scores


class FeedForward(nn.Module):
    def __init__(self, config: VitConfig):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.model_dims, config.intermediate_dims),
            nn.GELU(),
            nn.Linear(config.intermediate_dims, config.model_dims),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class VitBlock(nn.Module):
    def __init__(self, config: VitConfig, scale: int = 2):
        super(VitBlock, self).__init__()
        self.attn = Attention(config=config, is_cross_attn=False)
        self.config = config
        self.scale = scale
        self.lin_1 = nn.Linear(config.model_dims * scale, config.model_dims)
        self.lin_2 = nn.Linear(config.model_dims, config.model_dims * scale)
        self.norm = nn.LayerNorm(config.model_dims)
        self.ff = FeedForward(config=config)

    def forward(self, x):
        cls_tokens = x[:, :1, :]
        q = x[:, 1:, :]
        b, n, c = q.shape
        q = q.reshape(b, n//self.scale, c*self.scale)
        cls_tokens = self.lin_2(cls_tokens)
        q = torch.concat([cls_tokens, q], dim=1)
        q = self.lin_1(q)

        y, score = self.attn(q, x, x)
        y = self.norm(y + q)
        y = self.ff(y) + y
        return y, score


class PatchEmbeddingLayer(nn.Module):
    def __init__(self, config: VitConfig):
        super(PatchEmbeddingLayer, self).__init__()
        self.patch_conv = nn.Conv2d(
            config.input_channels,
            out_channels=config.model_dims,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        self.num_tokens = ((config.image_size//config.patch_size)**2) + 1

        self.cls_emb = nn.Parameter(torch.randn(1, 1, config.model_dims))
        self.pos_emb = nn.Parameter(torch.randn(
            1, self.num_tokens, config.model_dims))

    def forward(self, x):
        x = self.patch_conv(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        cls_emb = self.cls_emb.expand(b, 1, -1)
        x = torch.concat([x, cls_emb], dim=1)

        x = x + self.pos_emb
        return x


class ViT(nn.Module):
    def __init__(self, config: VitConfig):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbeddingLayer(config=config)
        self.vit_blocks = nn.ModuleList(
            [
                VitBlock(config=config, scale=1),
                VitBlock(config=config, scale=1),

                VitBlock(config=config, scale=2),
                VitBlock(config=config, scale=1),
                VitBlock(config=config, scale=1),
                VitBlock(config=config, scale=1),

                VitBlock(config=config, scale=2),
                VitBlock(config=config, scale=1),
                VitBlock(config=config, scale=1),
                VitBlock(config=config, scale=1),
                VitBlock(config=config, scale=1),

                VitBlock(config=config, scale=2),
                VitBlock(config=config, scale=1),
                VitBlock(config=config, scale=1),
                VitBlock(config=config, scale=1),
                VitBlock(config=config, scale=1),
                VitBlock(config=config, scale=1),
            ]
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        hidden_states = []
        attention_scores = []

        for layer in self.vit_blocks:
            x, attention_score = layer(x)
            hidden_states.append(x)
            attention_scores.append(attention_score)

        return ViTOutput(
            last_hidden_state=x,
            hidden_states=hidden_states,
            attention_scores=attention_scores
        )


class ViTModel(PreTrainedModel):
    config_class = VitConfig

    def __init__(self, config: VitConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.vit = ViT(config=config)

    def forward(self, x):
        return self.vit(x)
