import argparse
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from network import TransformerNetwork
from tokenizers import PAD_IDX, SpacyTokenizer
from torchtext.datasets import multi30k, Multi30k

class Trainer:
    def __init__(self, *args, **kwargs):
        self.model = TransformerNetwork(*args, **kwargs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        self.loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.src_tokenizer = SpacyTokenizer(lang="en_core_web_sm")
        self.tgt_tokenizer = SpacyTokenizer(lang="de_core_news_sm")

    def _collate_fn(self, batch: List[torch.Tensor]) -> torch.Tensor:
        src_batch, tgt_batch = [], []
        # Tokenizes the text, converts tokens to ids, and adds BOS/EOS tokens
        for src_sample, tgt_sample in batch:
            src_batch.append(self.src_tokenizer.tokenize(src_sample.strip("\n"), torch=True))
            tgt_batch.append(self.tgt_tokenizer.tokenize(tgt_sample.strip("\n"), torch=True))
        # Padding sequences to be the same length
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    def train(self):
        train_iter = Multi30k(split="train", language_pair=("en", "de"))
        data_loader = DataLoader(train_iter, batch_size=32, collate_fn=self._collate_fn)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for epoch in range(10):
            for src_batch, target_batch in data_loader:
                src_batch = src_batch.to(device)

                self.optimizer.zero_grad()
                output = self.model(src_batch, target_batch[:-1, :])
                # TODO: need to do some masking here
                loss = self.loss(output.reshape(-1, output.shape[-1]), target_batch[1:, :].reshape(-1))
                loss.backward()
                self.optimizer.step()
                print(loss.item())
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
