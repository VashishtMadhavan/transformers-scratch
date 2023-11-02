import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, num_layers: int = 6):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.encoder_stack = nn.ModuleList([EncoderLayer() for _ in range(self.num_layers)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for encoder_layer in self.encoder_stack:
            x = encoder_layer(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, num_layers: int = 6):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.decoder_stack = nn.ModuleList([DecoderLayer() for _ in range(self.num_layers)])
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        for decoder_layer in self.decoder_stack:
            x = decoder_layer(x, encoder_output)
        return x


class TransformerNetwork(nn.Module):
    def __init__(self, dim: int = 64, vocab_size: int = 256) -> None:
        self.dim = dim
        self.vocab_size = vocab_size
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_layer = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Take in a src/tgt sequence
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(y, encoder_output)
        return self.output_layer(decoder_output)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, y: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.decoder(y, encoder_output)