import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Any, List
from torchtext.datasets import Multi30k
from tokenizers import PAD_IDX
from utils import create_mask
from torch.utils.data import DataLoader


class TranslationDataset:
    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        src_tokenizer: Any,
        tgt_tokenizer: Any,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_tokenizer = self.src_tokenizer.build_vocab(
            self.get_data_iter(split="train"),
            index=0,
        )
        self.tgt_tokenizer = self.tgt_tokenizer.build_vocab(
            self.get_data_iter(split="train"),
            index=1,
        )

    def get_data_iter(self, split: str = "train") -> Any:
        return Multi30k(split=split, language_pair=(self.src_lang, self.tgt_lang))

    def get_data_loader(self, batch_size: int = 32, split: str = "train") -> DataLoader:
        return DataLoader(
            self.get_data_iter(split=split),
            batch_size=batch_size,
            collate_fn=self._collate_fn,
        )

    def get_vocab_size(self, type: str = "src") -> int:
        return (
            self.src_tokenizer.get_vocab_size()
            if type == "src"
            else self.tgt_tokenizer.get_vocab_size()
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

        # Apply Mask
        src_mask, tgt_mask = create_mask(src_batch, tgt_batch)
        return src_batch, tgt_batch, src_mask, tgt_mask
