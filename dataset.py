import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Any, List
from torchtext.datasets import Multi30k
from tokenizers import PAD_IDX

from torch.utils.data import DataLoader


class Dataset:
    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        src_tokenizer: Any,
        tgt_tokenizer: Any,
        split: str = "train",
        batch_size: int = 32,
    ):
        self.data_iter = Multi30k(split=split, language_pair=(src_lang, tgt_lang))
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.dataloader = DataLoader(
            self.data_iter, batch_size=batch_size, collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch: List[torch.Tensor]) -> torch.Tensor:
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(
                self.src_tokenizer.tokenize(src_sample.strip("\n"), torch=True)
            )
            tgt_batch.append(
                self.tgt_tokenizer.tokenize(tgt_sample.strip("\n"), torch=True)
            )

        # Padding Sequence
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    @staticmethod
    def create_mask(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src and target are seq_len x batch_size
        # The mask is 1 where we attend and 0 where we ignore
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        # Masking all tokens after a given token in the target sequence
        src_mask = torch.ones((1, src_seq_len, src_seq_len)).type(torch.bool)
        # Create a lower triangular matrix
        tgt_mask = Dataset.subsequent_mask(tgt_seq_len)

        # Mask padding tokens
        src_padding_mask = (
            (src != PAD_IDX).transpose(0, 1).unsqueeze(-2)
        )  # batch_size x 1 x seq_len
        tgt_padding_mask = (
            (tgt != PAD_IDX).transpose(0, 1).unsqueeze(-2)
        )  # batch_size x 1 x seq_len

        # Combine padding masks with subsequent mask
        tgt_mask = tgt_mask & tgt_padding_mask
        src_mask = src_mask & src_padding_mask
        return src_mask, tgt_mask

    @staticmethod
    def subsequent_mask(tgt_seq_len: int) -> torch.Tensor:
        return (
            torch.tril(torch.ones((1, tgt_seq_len, tgt_seq_len))) == 1
        )  # 1 x seq_len x seq_len
