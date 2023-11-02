import torch
import torch.nn as nn
import numpy as np
import math

# Embeds each token in vocab into vector space. Simple lookup table.
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 256, dim: int = 64) -> None:
        super(TokenEmbedding, self).__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * np.sqrt(self.dim)

# Captures information about word order when passing tokens through self-attention
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
