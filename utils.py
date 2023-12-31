import torch
from tokenizers import PAD_IDX


def subsequent_mask(tgt_seq_len: int) -> torch.Tensor:
    return (
        torch.tril(torch.ones((1, tgt_seq_len, tgt_seq_len))) == 1
    )  # 1 x seq_len x seq_len


def create_mask(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    # src and target are seq_len x batch_size
    # The mask is 1 where we attend and 0 where we ignore
    tgt_seq_len = tgt.shape[0]
    # Create a lower triangular matrix
    tgt_mask = subsequent_mask(tgt_seq_len)

    # Mask padding tokens
    src_padding_mask = (
        (src != PAD_IDX).transpose(0, 1).unsqueeze(-2)
    )  # batch_size x 1 x seq_len
    tgt_padding_mask = (
        (tgt != PAD_IDX).transpose(0, 1).unsqueeze(-2)
    )  # batch_size x 1 x seq_len

    # Combine padding masks with subsequent mask
    tgt_mask = tgt_padding_mask & tgt_mask  # batch_size x seq_len x seq_len
    return src_padding_mask, tgt_mask
