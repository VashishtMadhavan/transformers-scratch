import torch
import torch.nn as nn
import numpy as np

# Generalized Query Attention where K,V heads are split into groups. MQA is with groups=1 and MHA is groups=num_heads
class GroupQueryAttention(nn.Module):
    def __init__(
        self, input_dim: int, dim: int = 64, num_heads: int = 8, groups: int = 1
    ):
        super(GroupQueryAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.groups = groups
        if self.num_heads % self.groups != 0:
            raise ValueError("Number of heads must be divisible by number of groups")

        # Linear layers for query
        self.Q = nn.ModuleList(
            [nn.Linear(input_dim, self.dim) for _ in range(self.num_heads)]
        )
        # Shared K,V for all heads
        self.K = nn.ModuleList(
            [nn.Linear(input_dim, self.dim) for _ in range(self.groups)]
        )
        self.V = nn.ModuleList(
            [nn.Linear(input_dim, self.dim) for _ in range(self.groups)]
        )
        self.W = nn.Linear(num_heads * self.dim, self.dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x is (batch_size, seq_len, input_dim)
        z_list = []
        for i in range(self.num_heads):
            # Applying attention to each head
            K = self.K[i % self.groups](key)
            V = self.V[i % self.groups](value)
            Q = self.Q[i](query)
            attn = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(
                self.dim
            )
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)
            z = torch.bmm(torch.softmax(attn, dim=-1), V)
            z_list.append(z)
        z_total = torch.cat(z_list, dim=-1)  # (batch_size, seq_len, num_heads * dim)
        return z_total


# Implementing a multi-headed attention layer
class MultiHeadedAttention(GroupQueryAttention):
    def __init__(self, input_dim: int, dim: int = 64, num_heads: int = 8):
        super().__init__(input_dim, dim, num_heads, groups=num_heads)


# Implementing a multi-query attention layer
class MultiQueryAttention(GroupQueryAttention):
    def __init__(self, input_dim: int, dim: int = 64, num_heads: int = 8):
        super().__init__(input_dim, dim, num_heads, groups=1)
