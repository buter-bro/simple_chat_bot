import torch

from model.layers import DecoderLayer
from torch import nn
from model.embeddings import Embeddings


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.layers_num)])

    def forward(self, decoder_inputs: torch.Tensor, decoder_mask: torch.Tensor = None) -> torch.Tensor:
        outputs = decoder_inputs
        for decoder_layer in self.layers:
            outputs = decoder_layer(outputs, decoder_mask)
        return outputs


class TransformerOutput(nn.Module):

    def __init__(self, config):
        super(TransformerOutput, self).__init__()
        self.output_feed_forward = nn.Linear(config.d_model, config.vocabulary_size - 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_feed_forward.weight)
        nn.init.zeros_(self.output_feed_forward.bias)

    def forward(self, inputs):
        return self.output_feed_forward(inputs)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.embeddings = Embeddings(config.model)
        self.decoder = Decoder(config.model)
        self.output = TransformerOutput(config.model)

    def forward(self, decoder_inputs: torch.Tensor, decoder_mask: torch.Tensor = None) -> torch.Tensor:
        embedded_inputs = self.embeddings(decoder_inputs)
        decoder_output = self.decoder(embedded_inputs, decoder_mask)
        output = self.output(decoder_output)
        return output



