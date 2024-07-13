from torch import nn
import torch


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(config.vocabulary_size, config.d_model, padding_idx=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embeddings(inputs)





