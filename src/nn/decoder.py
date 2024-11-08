from torch.nn import functional as nnf
from typing import Optional
from torch import nn
import torch

class TokenEmbedding(nn.Embedding):
    def __init__(
        self,
        vocab_size : int,
        embed_size : int,
        pad_idx : int = 0
    ):
        super(TokenEmbedding, self).__init__(
            vocab_size,
            embed_size,
            padding_idx = pad_idx
        )

class Embedding(nn.Module):
    def __init__(
        self,
        cfg,
    ) -> None:
        super(Embedding, self).__init__()
        self.n_embd = cfg.n_embd
        self.token = TokenEmbedding(
            vocab_size = cfg.n_vocab,
            embed_size = self.n_embd,
            pad_idx = cfg.pad_idx
        )
        self.position = nn.Embedding(
            cfg.max_positions,
            self.n_embd
        )

        self.norm = nn.LayerNorm(self.n_embd, eps = cfg.norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "position_ids",
            torch.arange(cfg.max_positions).expand((1, -1)),
            persistent = False
        )

    def forward(
        self,
        sequence : torch.LongTensor
    ) -> torch.FloatTensor:
        input_embd = self.token(sequence)

        positions = self.position(self.position_ids[:, :input_embd.size(1)])

        embd = input_embd + positions

        return self.dropout(self.norm(embd))

class SelfAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim = cfg.n_embd,
            num_heads = cfg.n_head,
            dropout = cfg.dropout,
            batch_first = True,
            bias = cfg.qkv_bias
        )

    def forward(
        self,
        query : torch.FloatTensor,
        kv_cross : torch.FloatTensor,
        padding_mask : Optional[torch.BoolTensor] = None,
        att_mask : Optional[torch.BoolTensor] = None,
        return_attn_weights : bool = False
    ) -> torch.FloatTensor:
        att, w = self.multihead_attn(
            query,
            kv_cross,
            kv_cross,
            attn_mask = att_mask,
            key_padding_mask = padding_mask
        )
        if return_attn_weights:
            return (att, w)
        return att

class Attention(nn.Module):
    def __init__(self, cfg) -> None:
        super(Attention, self).__init__()
        self.attnetion = SelfAttention(cfg)
        self.projection = nn.Linear(cfg.n_embd, cfg.n_embd)
    
    def forward(
        self,
        query : torch.FloatTensor,
        kv_cross : torch.FloatTensor,
        padding_mask : Optional[torch.BoolTensor] = None,
        att_mask : Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        attnetion_output = self.attnetion(
            query = query,
            kv_cross = kv_cross,
            padding_mask = padding_mask,
            att_mask = att_mask,
        )
        return self.projection(attnetion_output)

class FeedForward(nn.Sequential):
    def __init__(self, cfg) -> None:
        super(FeedForward, self).__init__(
            nn.Linear(cfg.n_embd, cfg.n_embd * cfg.ffn_ratio),
            nn.GELU(),
            nn.Linear(cfg.n_embd * cfg.ffn_ratio, cfg.n_embd)
        )

class DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super(DecoderLayer, self).__init__()
        self.att_norm_1 = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        self.attn_1 = Attention(config)

        self.att_norm_2 = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        self.attn_2 = Attention(config)

        self.ffn_norm = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        self.ffn = FeedForward(config)

        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        encoder_hidden_states : torch.FloatTensor,
        hidden_states : torch.FloatTensor,
        padding_mask : Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        _, l, _ = hidden_states.shape
        att_mask = torch.triu(
            torch.ones(l, l,device = hidden_states.device), 1
        ).bool()
        
        hidden_states_norm = self.att_norm_1(hidden_states)
        attention = self.attn_1(
            query = hidden_states_norm,
            kv_cross = hidden_states_norm,
            padding_mask = padding_mask,
            att_mask = att_mask
        )
        hidden_states = hidden_states + self.drop(attention)

        hidden_states_norm = self.att_norm_2(hidden_states)
        attention = self.attn_2(
            query = hidden_states_norm,
            kv_cross = encoder_hidden_states,
        )
        hidden_states = hidden_states + self.drop(attention)
        
        hidden_states_norm = self.ffn_norm(hidden_states)
        ffn_out = self.ffn(hidden_states_norm)
        hidden_states = hidden_states + self.drop(ffn_out)
        return hidden_states

class TransformerDecoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransformerDecoder, self).__init__()

        self.transformer_enc = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.n_decoder_layer)]
        )
        self.norm = nn.LayerNorm(config.n_embd, eps = config.norm_eps)
        # self.norm = nn.Identity()

    def forward(
        self,
        encoder_hidden_states : torch.FloatTensor,
        input_embedding : torch.FloatTensor,
        input_padding_mask : Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        hidden_states = input_embedding
        for block in self.transformer_enc:
            hidden_states = block(
                encoder_hidden_states = encoder_hidden_states,
                hidden_states = hidden_states,
                padding_mask = input_padding_mask,
            )
        return self.norm(hidden_states)

class Decoder(nn.Module):
    def __init__(self, cfg) -> None:
        super(Decoder, self).__init__()
        self.pad_idx = cfg.pad_idx
        self.emb = Embedding(cfg)
        self.trdec = TransformerDecoder(cfg)
        self.out = nn.Linear(
            cfg.n_embd,
            cfg.n_vocab,
            bias = False
        )

    def forward(
        self,
        encoder_hidden_states : torch.FloatTensor,
        input_ids : torch.LongTensor,
        input_padding_mask : Optional[torch.ByteTensor] = None,
    ) -> torch.FloatTensor:
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask == self.pad_idx

        input_embedding = self.emb(input_ids)
        hidden_states = self.trdec(
            encoder_hidden_states = encoder_hidden_states,
            input_embedding = input_embedding,
            input_padding_mask = input_padding_mask,
        )
        return self.out(hidden_states)