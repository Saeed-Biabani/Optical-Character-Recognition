from torch.distributions import Categorical
from torch.nn import functional as nnf
from dataclasses import dataclass
from ..dataset import MetaData
from .encoder import Encoder
from .decoder import Decoder
from typing import Optional
from torch import nn
import torch

@dataclass
class TRoutput:
    encoder_hidden_states : Optional[torch.FloatTensor] = None
    loss : Optional[torch.FloatTensor] = None
    logits : torch.FloatTensor = None

@dataclass
class TRconfig:
    initializer_range : float = 0.02
    image_size : tuple[int] = None
    patch_size : tuple[int] = None
    num_channels : int = None

    n_encoder_layer : int = 4
    n_decoder_layer : int = 4

    max_positions : int = None
    qkv_bias : bool = False
    n_embd : int = 256
    norm_eps : float = 1e-6
    dropout : float = 0.1
    n_head : int = 8
    ffn_ratio : int = 4

    n_vocab : int = None

    pad_idx : int = None
    sos_idx : int = None
    eos_idx : int = None

class TRnet(nn.Module):
    def __init__(self, cfg : TRconfig) -> None:
        super(TRnet, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def read_image(
        self,
        pixel_values : torch.FloatTensor,
        temp : float = 0.1,
        device : str = 'cuda',
        max_generation : int = 64
    ) -> torch.LongTensor:
        sos_token = self.cfg.sos_idx * torch.ones(1, 1).long()
        log_tokens = [sos_token]

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                encoder_hidden_states = self.apply_encoder(
                    pixel_values = pixel_values.to(device)
                )

            for _ in range(max_generation):
                input_tokens = torch.cat(log_tokens, 1)

                data_pred = self.apply_decoder(
                    encoder_hidden_states = encoder_hidden_states,
                    input_ids = input_tokens.to(device),
                )

                dist = Categorical(logits = data_pred[:, -1] / temp)
                next_tokens = dist.sample().reshape(1, 1)

                log_tokens.append(next_tokens.cpu())

                if next_tokens.item() == self.cfg.eos_idx:
                    break

        return torch.cat(log_tokens, 1).cpu().detach()
    
    def apply_encoder(
        self,
        pixel_values : torch.FloatTensor,
    ) -> torch.FloatTensor:
        return self.encoder(
            pixel_values = pixel_values
        )

    def apply_decoder(
        self,
        encoder_hidden_states : torch.FloatTensor,
        input_ids : torch.LongTensor,
        input_padding_mask : Optional[torch.ByteTensor] = None,
    ) -> torch.FloatTensor:
        return self.decoder(
            encoder_hidden_states = encoder_hidden_states,
            input_ids = input_ids,
            input_padding_mask = input_padding_mask,
        )

    def forward(
        self,
        pixel_values : torch.FloatTensor,
        input_ids : torch.LongTensor,
        input_padding_mask : Optional[torch.ByteTensor] = None,
        labels : torch.LongTensor = None
    ) -> TRoutput:
        encoder_hidden_states = self.encoder(
            pixel_values = pixel_values,
        )
        decoder_output = self.decoder(
            encoder_hidden_states = encoder_hidden_states,
            input_ids = input_ids,
            input_padding_mask = input_padding_mask,
        )

        loss = None
        if labels is not None:
            loss = (nnf.cross_entropy(
                decoder_output.transpose(1, 2),
                labels, reduction = 'none'
            ) * input_padding_mask).mean()

        return TRoutput(
            loss = loss,
            encoder_hidden_states = encoder_hidden_states,
            logits = decoder_output
        )

    @staticmethod
    def from_pretrained(
        path : str,
        device : str = 'cpu',
        return_metadata : bool = True
    ) -> nn.Module:
        data = torch.load(
            path,
            weights_only = False,
            map_location = device
        )
        model = TRnet(
            TRconfig(
                **data['model_config']
            )
        )
        model.load_state_dict(
            data['model_weights']
        )
        model.eval()
        if return_metadata:
            return (model, MetaData(**data['metadata']))
        return model