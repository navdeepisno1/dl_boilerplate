import torch.nn as nn
import torch
from transformers import PretrainedConfig, PreTrainedModel


class CausalLoss:
    def __call__(self, labels, logits):
        loss_fn = nn.CrossEntropyLoss()

        loss = 0
        for i,logit in enumerate(logits):
            idx = i+1
            shifted_labels = labels[:,idx:]
            shifted_logits = logit[:,:-idx,:]    
            loss = loss + loss_fn(shifted_logits.reshape(-1,shifted_logits.shape[-1]),shifted_labels.reshape(-1))

        return loss


class GPTNaviOutput:
    def __init__(
        self,
        logits,
        last_hidden_state,
        hidden_states
    ):
        self.logits = logits
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class GPTNaviConfig(PretrainedConfig):
    model_type = 'GPTNavi'

    def __init__(
            self,
            hidden_size: int = 64,
            intermediate_size: int = 128,
            num_blocks: int = 16,
            num_heads: int = 8,
            vocab_size: int = 50257,
            max_pos_emb_tokens: int = 128,
            padding_idx: int = 50256,
            num_outputs:int = 5,
            **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_pos_emb_tokens = max_pos_emb_tokens
        self.padding_idx = padding_idx
        self.num_outputs = num_outputs


class GPTNaviAttention(nn.Module):
    def __init__(self, config: GPTNaviConfig):
        super(GPTNaviAttention, self).__init__()

        self.num_heads = config.num_heads
        self.head_dims = config.hidden_size // config.num_heads

        self.scale = self.head_dims ** -0.5

        self.to_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.to_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.to_v = nn.Linear(config.hidden_size, config.hidden_size)

        self.to_o = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, q, k, v, attention_mask):
        num_heads = self.num_heads
        head_dims = self.head_dims
        b, n, c = q.shape

        q = self.to_q(q).reshape(b, n, num_heads,
                                 head_dims).permute(0, 2, 1, 3)
        k = self.to_q(k).reshape(b, n, num_heads,
                                 head_dims).permute(0, 2, 1, 3)
        v = self.to_q(v).reshape(b, n, num_heads,
                                 head_dims).permute(0, 2, 1, 3)

        scores = q@k.permute(0, 1, 3, 2)
        scores = scores * self.scale
        scores = scores + attention_mask
        scores = nn.Softmax(dim=-1)(scores)

        x = scores@v
        x = x.permute(0, 2, 1, 3).reshape(b, n, c)
        x = self.to_q(x)
        return x


class GPTNaviMLP(nn.Module):
    def __init__(self, config: GPTNaviConfig):
        super(GPTNaviMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class GPTNaviBlock(nn.Module):
    def __init__(self, config: GPTNaviConfig):
        super(GPTNaviBlock, self).__init__()

        self.norm_1 = nn.LayerNorm(config.hidden_size)
        self.attn = GPTNaviAttention(config=config)
        self.norm_2 = nn.LayerNorm(config.hidden_size)

        self.ff = GPTNaviMLP(config=config)

    def forward(self, x, attention_mask):
        residual = x
        x = self.norm_1(x)
        x = self.attn(
            q=x,
            k=x,
            v=x,
            attention_mask=attention_mask
        )
        x = x + residual

        residual = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = x + residual

        return x


class GPTNaviModel(nn.Module):
    def __init__(self, config: GPTNaviConfig):
        super(GPTNaviModel, self).__init__()
        self.config = config
        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.padding_idx
        )
        self.pos_emb = nn.Embedding(
            config.max_pos_emb_tokens+1,
            config.hidden_size,
            padding_idx=config.max_pos_emb_tokens
        )
        
        self.norm = nn.ModuleList([nn.LayerNorm(config.hidden_size)]*config.num_outputs)

        self.transformer_blocks = nn.ModuleList(
            [GPTNaviBlock(config=config) for _ in range(config.num_blocks)]
        )



        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.lm_head.weight = self.token_emb.weight

    def get_pos_embeds(self, attention_mask):
        pos_tokens = torch.cumsum(attention_mask, 1)
        pos_tokens[attention_mask == 0] = self.config.max_pos_emb_tokens
        pos_embeds = self.pos_emb(pos_tokens.to(torch.int32))

        pos_embeds = pos_embeds.to(attention_mask.device)
        return pos_embeds

    def get_causal_mask(self, attention_mask):
        seq_size = attention_mask.shape[1]
        mask = torch.ones(seq_size, seq_size)
        mask = torch.tril(mask)
        mask[mask == 0] = float('-inf')
        mask[mask == 1] = 0

        mask = mask.to(attention_mask.device)
        return mask

    def forward(self, input_ids=None, attention_mask=None, input_embeds=None) -> GPTNaviOutput:
        assert input_ids is not None or input_embeds is not None, "Provide one of input_ids and input_embeds"
        if input_embeds is None:
            input_embeds = self.token_emb(input_ids)

        pos_embeds = self.get_pos_embeds(attention_mask=attention_mask)
        hidden_state = input_embeds + pos_embeds

        attention_mask = self.get_causal_mask(attention_mask=attention_mask)

        hidden_states = []
        for block in self.transformer_blocks:
            hidden_state = block(hidden_state, attention_mask)
            hidden_states.append(hidden_state)

        logits = []
        for i in range(-1*self.config.num_outputs,0,1):
            logit = self.norm[i](hidden_states[i])
            logit = self.lm_head(logit)
            logits.append(logit)


        return GPTNaviOutput(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
            logits=logits
        )
