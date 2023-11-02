import torch
import torch.nn as nn
from blocks import EncoderLayer, DecoderLayer
from embeddings import TokenEmbedding, PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(Encoder, self).__init__()
        self.encoder_stack = nn.ModuleList(
            [
                EncoderLayer(dim=dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for encoder_layer in self.encoder_stack:
            x = encoder_layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(Decoder, self).__init__()
        self.decoder_stack = nn.ModuleList(
            [
                DecoderLayer(dim=dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        for decoder_layer in self.decoder_stack:
            x = decoder_layer(x, encoder_output, src_mask, target_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        src_vocab_size: int = 256,
        target_vocab_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            num_layers=num_layers, dim=dim, num_heads=num_heads, dropout=dropout
        )
        self.decoder = Decoder(
            num_layers=num_layers, dim=dim, num_heads=num_heads, dropout=dropout
        )
        # Outputs tokens in the vocab
        self.output_layer = nn.Linear(dim, target_vocab_size)

        # Getting input embedding
        self.src_embed = nn.Sequential(
            TokenEmbedding(src_vocab_size, dim), PositionalEncoding(dim)
        )
        self.target_embed = nn.Sequential(
            TokenEmbedding(target_vocab_size, dim), PositionalEncoding(dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        src_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x and y are src and target sequences respectively
        z = self.encode(x, src_mask)
        y_hat = self.decode(y, z, src_mask, target_mask)
        return self.output_layer(y_hat)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_emb = self.src_embed(x)
        return self.encoder(x_emb, mask)

    def decode(
        self,
        y: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        y_emb = self.target_embed(y)
        return self.decoder(y_emb, enc_output, src_mask, target_mask)
