import torch
import torch.nn as nn
from attention import MultiHeadedAttention, MultiQueryAttention

class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p = dropout

        self.feed_forward = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Dropout(self.p),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x)

class AddNorm(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(AddNorm, self).__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(self.input_dim)
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        return x + self.drop(module(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, dim: int = 64):
        super(EncoderLayer, self).__init__()
        self.dim = dim
        self.attention = MultiHeadedAttention(input_dim=dim)
        self.add_norm1 = AddNorm(dim)
        self.feed_forward = FeedForward(input_dim=dim, hidden_dim=dim)
        self.add_norm2 = AddNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.add_norm1(x, self.attention)
        z2 = self.add_norm2(z1, self.feed_forward)
        return z2
    
class DecoderLayer(nn.Module):
    def __init__(self, dim: int = 64):
        super(DecoderLayer, self).__init__()
        self.dim = dim
        self.attention = MultiHeadedAttention(input_dim=dim)
        self.add_norm1 = AddNorm(dim)

        self.enc_attention = MultiHeadedAttention(input_dim=dim)
        self.add_norm2 = AddNorm(dim)

        self.feed_forward = FeedForward(input_dim=dim, hidden_dim=dim)
        self.add_norm3 = AddNorm(dim)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        z1 = self.add_norm1(x, self.attention)
        z2 = self.add_norm2(z1, self.enc_attention)
        z3 = self.add_norm3(z2, self.feed_forward)
        return z3