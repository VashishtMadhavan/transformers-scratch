import torch
import torch.nn as nn
from attention import MultiHeadedAttention

class FeedForward(nn.Module):
    def __init__(self, input_dim: int, dim: int = 2048, dropout: float = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int = 64, num_heads: int = 8, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadedAttention(input_dim=dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        self.feed_forward = FeedForward(input_dim=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm(x)
        h1 = self.attention(norm_x, norm_x, norm_x, mask)
        z1 = x + self.drop(h1)

        norm_z1 = self.norm2(z1)
        h2 = self.feed_forward(norm_z1)
        z2 = z1 + self.drop2(h2)
        return z2
    
class DecoderLayer(nn.Module):
    def __init__(self, dim: int = 64, num_heads: int = 8, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.dim = dim
        self.attention = MultiHeadedAttention(input_dim=dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)

        self.enc_attention = MultiHeadedAttention(input_dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

        self.feed_forward = FeedForward(input_dim=dim, hidden_dim=dim)
        self.norm3 = nn.LayerNorm(dim)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        h1 = self.attention(norm_x, norm_x, norm_x, target_mask)
        z1 = x + self.drop1(h1)

        norm_z1 = self.norm2(z1)
        # Getting the query from the decoder and key, value from the encoder
        h2 = self.enc_attention(norm_z1, enc_output, enc_output, src_mask)
        z2 = z1 + self.drop2(h2)

        norm_z2 = self.norm3(z2)
        h3 = self.feed_forward(norm_z2)
        z3 = z2 + self.drop3(h3)
        return z3
