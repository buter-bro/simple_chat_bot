import torch

from model.layers import DecoderLayer
from torch import nn
from model.embeddings import Embeddings


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.layers_num)])

    def forward(self, decoder_inputs: torch.Tensor, positional_encoding, decoder_mask: torch.Tensor = None) -> torch.Tensor:
        outputs = decoder_inputs
        for decoder_layer in self.layers:
            outputs = decoder_layer(outputs, positional_encoding, decoder_mask)
        return outputs


class TransformerOutput(nn.Module):

    def __init__(self, config, decoder_vocabulary_size):
        super(TransformerOutput, self).__init__()
        self.output_feed_forward = nn.Linear(
            config.d_model,
            decoder_vocabulary_size - 1,
            bias=False
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_feed_forward.weight)

    def forward(self, inputs):
        return self.output_feed_forward(inputs)


class Transformer(nn.Module):
    def __init__(self, config, decoder_vocabulary_size):
        super(Transformer, self).__init__()

        self.embeddings = Embeddings(config.model, decoder_vocabulary_size)
        self.embeddings_dropout = nn.Dropout(p=config.model.dropout_rate)
        self.decoder = Decoder(config.model)
        self.final_layer_norm = nn.LayerNorm(config.model.d_model)
        self.output = TransformerOutput(config.model, decoder_vocabulary_size)

    def forward(self, decoder_inputs: torch.Tensor, positional_encoding, decoder_mask: torch.Tensor = None) -> torch.Tensor:
        embedded_inputs = self.embeddings_dropout(self.embeddings(decoder_inputs))
        decoder_output = self.decoder(embedded_inputs, positional_encoding, decoder_mask)
        output = self.output(self.final_layer_norm(decoder_output))
        return output



