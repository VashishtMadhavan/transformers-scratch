import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from network import Transformer
from tokenizers import PAD_IDX, SpacyTokenizer


class Trainer:
    def __init__(self, dataset, *args, **kwargs):
        self.model = Transformer(*args, **kwargs)
        self.dataset = dataset
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.src_tokenizer = SpacyTokenizer(lang="en_core_web_sm")
        self.tgt_tokenizer = SpacyTokenizer(lang="de_core_news_sm")

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_loader = self.dataset.train_loader

        for epoch in range(10):
            for src_batch, target_batch in data_loader:
                src_batch = src_batch.to(device)
                target_batch = target_batch.to(device)

                src_mask, target_mask = self.dataset.create_mask(
                    src_batch, target_batch
                )

                self.optimizer.zero_grad()
                output = self.model(
                    src_batch, target_batch[:-1, :], src_mask, target_mask
                )
                # TODO: need to do some masking here
                loss = self.loss(
                    output.reshape(-1, output.shape[-1]),
                    target_batch[1:, :].reshape(-1),
                )
                loss.backward()
                self.optimizer.step()
                print(loss.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    dataset = 
