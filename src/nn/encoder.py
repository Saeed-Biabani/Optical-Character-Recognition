from einops.layers.torch import Rearrange
from torch.nn import functional as nnf
from typing import Union
from torch import nn
import torch
import math

class PatchEmbedding(nn.Module):
    def __init__(self, cfg) -> None:
        super(PatchEmbedding, self).__init__()
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.num_channels = cfg.num_channels
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        self.projection = nn.Conv2d(
            in_channels = cfg.num_channels,
            out_channels = cfg.n_embd,
            kernel_size = self.patch_size,
            stride = self.patch_size
        )
        self.reshape = Rearrange('b c h w -> b (h w) c')

    def forward(
        self,
        pixel_values : torch.FloatTensor
    ) -> torch.FloatTensor:
        embeddings = self.projection(pixel_values)
        return self.reshape(embeddings)

class Embedding(nn.Module):
    def __init__(self, cfg) -> None:
        super(Embedding, self).__init__()
        self.patch_size = cfg.patch_size
        self.patch_extractor = PatchEmbedding(cfg)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, cfg.n_embd)
        )

        num_patches = self.patch_extractor.num_patches + 1
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches, cfg.n_embd)
        )

        self.norm = nn.LayerNorm(cfg.n_embd, eps = cfg.norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def interpolate_pos_encoding(
        self,
        embeddings : torch.FloatTensor,
        height : int,
        width : int
    ) -> torch.FloatTensor:
        num_positions = self.position_embeddings.shape[1] - 1

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values : torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size, _, height, width = pixel_values.shape

        embeddings = self.patch_extractor(pixel_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim = 1)

        positions = self.interpolate_pos_encoding(embeddings, height, width)
        embeddings = embeddings + positions

        return self.dropout(self.norm(embeddings))

class SelfAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super(SelfAttention, self).__init__()
        
        self.num_attention_heads = cfg.n_head
        self.attention_head_size = int(cfg.n_embd / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.n_embd, self.all_head_size, bias = cfg.qkv_bias)
        self.key = nn.Linear(cfg.n_embd, self.all_head_size, bias = cfg.qkv_bias)
        self.value = nn.Linear(cfg.n_embd, self.all_head_size, bias = cfg.qkv_bias)

        self.dropout = nn.Dropout(cfg.dropout)

    def transpose_for_scores(self, x: torch.FloatTensor) -> torch.FloatTensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states : torch.FloatTensor
    ) -> torch.FloatTensor:

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return context_layer

class Attention(nn.Sequential):
    def __init__(self, cfg) -> None:
        super(Attention, self).__init__(
            SelfAttention(cfg),
            nn.Linear(cfg.n_embd, cfg.n_embd)
        )

class FeedForward(nn.Sequential):
    def __init__(self, cfg) -> None:
        super(FeedForward, self).__init__(
            nn.Linear(cfg.n_embd, cfg.n_embd * cfg.ffn_ratio),
            nn.GELU(),
            nn.Linear(cfg.n_embd * cfg.ffn_ratio, cfg.n_embd)
        )


class EncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super(EncoderLayer, self).__init__()
        self.att_norm = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        self.attn = Attention(config)

        self.ffn_norm = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        self.ffn = FeedForward(config)

        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states : torch.FloatTensor,
    ) -> torch.FloatTensor:
        hidden_states_norm = self.att_norm(hidden_states)
        attention = self.attn(
            hidden_states_norm
        )
        hidden_states = hidden_states + self.drop(attention)
        
        hidden_states_norm = self.ffn_norm(hidden_states)
        ffn_out = self.ffn(hidden_states_norm)
        hidden_states = hidden_states + self.drop(ffn_out)
        return hidden_states

class TransformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransformerEncoder, self).__init__()

        self.transformer_enc = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_encoder_layer)]
        )
        self.norm = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        # self.norm = nn.Identity()

    def forward(
        self,
        hidden_states : torch.FloatTensor,
    ) -> torch.FloatTensor:
        for block in self.transformer_enc:
            hidden_states = block(
                hidden_states = hidden_states
            )
        return self.norm(hidden_states)

class Encoder(nn.Module):
    def __init__(self, cfg) -> None:
        super(Encoder, self).__init__()
        self.config = cfg
        self.emb = Embedding(cfg)
        self.encoder = TransformerEncoder(cfg)

        # self.apply(self._init_weights)
    
    def _init_weights(
        self,
        module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]
    ) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embedding):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

    def forward(
        self,
        pixel_values : torch.FloatTensor,
    ) -> torch.FloatTensor:
        input_embedding = self.emb(pixel_values)
        return self.encoder(
            hidden_states = input_embedding
        )