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


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout_rate: float):
        super(RotaryPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.cos_rotation_matrix, self.sin_rotation_matrix = self.get_rotation_matrices(d_model, seq_len)

        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def get_rotation_matrices(d_model, seq_len):
        positions = torch.arange(0, seq_len, 1, dtype=torch.float32).unsqueeze(1)
        scaling_factor = torch.pow(torch.tensor(10000), torch.arange(0, d_model, 2) / d_model)

        cos_rotation_matrix = torch.zeros(seq_len, d_model)
        sin_rotation_matrix = torch.zeros(seq_len, d_model)
        cos_rotation_matrix[:, 0::2] = torch.cos(positions / scaling_factor)
        cos_rotation_matrix[:, 1::2] = torch.cos(positions / scaling_factor)
        sin_rotation_matrix[:, 0::2] = torch.sin(positions / scaling_factor)
        sin_rotation_matrix[:, 1::2] = torch.sin(positions / scaling_factor)

        return cos_rotation_matrix, sin_rotation_matrix

    @staticmethod
    def get_shifted_inputs(inputs: torch.Tensor):

        shifted_inputs = torch.zeros(inputs.shape)
        shifted_inputs[:, :, 1::2] = inputs[:, :, 0::2]
        shifted_inputs[:, :, 0::2] = -inputs[:, :, 1::2]
        return shifted_inputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rotated_inputs = self.cos_rotation_matrix * inputs + self.sin_rotation_matrix * self.get_shifted_inputs(inputs)
        return self.dropout(rotated_inputs)






