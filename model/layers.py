import torch
from torch import nn
from model.sublayers import MultiHeadAttention, FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.config = config

        self.mha_layer_norm = nn.LayerNorm(config.d_model)
        self.mha = MultiHeadAttention(config)
        self.mha_dropout = nn.Dropout(p=config.dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)
        self.ff_dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self, inputs: torch.Tensor, positional_encoding, decoder_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.mha_layer_norm(inputs)
        self_attention = self.mha_dropout(self.mha(x, x, x, positional_encoding, decoder_mask))
        mha_output = inputs + self_attention
        ff_output = self.ff_dropout(self.ff(self.ff_layer_norm(mha_output)))
        return ff_output + mha_output




