import torch
import torch.nn as nn
import numpy as np

class TokenEmbedding(nn.Module):
    def __init__(self, dim: int = 64, vocab_size: int = 256) -> None:
        super(TokenEmbedding, self).__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * np.sqrt(self.dim)
    

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position = torch.arange(0, x.size(1), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * -(np.log(10000.0) / self.dim))
        pos = position * div_term
        pos = pos.expand_as(x)
        return self.dropout(x + pos)