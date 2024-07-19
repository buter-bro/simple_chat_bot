import torch
from torch import nn
from model.modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.config = config
        self.d_k = self.config.d_model // self.config.heads_num
        self.sdpa = ScaledDotProductAttention(self.d_k, self.config.heads_num)

        self.weights_q = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.weights_k = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.weights_v = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.weights_o = nn.Linear(self.config.d_model, self.config.d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weights_q.weight)
        nn.init.xavier_uniform_(self.weights_k.weight)
        nn.init.xavier_uniform_(self.weights_v.weight)
        nn.init.xavier_uniform_(self.weights_o.weight)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                positional_encoding, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):

        weighted_queries = self.weights_q(queries)
        weighted_keys = self.weights_k(keys)
        weighted_values = self.weights_v(values)

        # Rotation positional encodings (RoPE)
        rotated_weighted_queries = positional_encoding(weighted_queries)
        rotated_weighted_keys = positional_encoding(weighted_keys)
        rotated_weighted_values = positional_encoding(weighted_values)

        attention = self.sdpa(rotated_weighted_queries, rotated_weighted_keys, rotated_weighted_values, mask)

        batch_size = queries.size(0)
        z = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.config.d_model)

        return self.weights_o(z)


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.config = config
        self.w_1 = nn.Linear(self.config.d_model, self.config.d_ff)
        self.w_2 = nn.Linear(self.config.d_ff, self.config.d_model)
        self.activation = getattr(torch.nn, self.config.activation)()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(inputs)))







