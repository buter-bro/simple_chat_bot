import torch
from torch import nn
from math import sqrt


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, heads_num: int):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.heads_num = heads_num

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:

        batch_size = queries.size(0)
        queries = queries.view(batch_size, -1, self.heads_num, self.d_k).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.heads_num, self.d_k).transpose(1, 2)
        values = values.view(batch_size, -1, self.heads_num, self.d_k).transpose(1, 2)

        attention = (queries @ keys.transpose(-2, -1)) / sqrt(self.d_k)
        if mask is not None:
            attention += (mask * torch.tensor(-1e9))
        attention_weights = nn.functional.softmax(attention, dim=-1)

        return attention_weights @ values




# class RotaryPositionalEncoding(nn.Module):
#     def __init__(self):
#         super(RotaryPositionalEncoding, self).__init__()
