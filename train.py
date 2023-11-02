import argparse
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim

from models import Transformer
from tokenizers import PAD_IDX, SpacyTokenizer
from dataset import TranslationDataset


class Trainer:
    def __init__(
        self,
        dataset: TranslationDataset,
        model: Transformer,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def train_epoch(self):
        model.train()
        train_loss = 0.0
        for src_batch, target_batch in self.dataset.train_loader:
            src_batch = src_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            src_mask, target_mask = self.dataset.create_mask(src_batch, target_batch)

            self.optimizer.zero_grad()
            logits = self.model(src_batch, target_batch[:-1, :], src_mask, target_mask)
            loss = self.loss(
                logits.reshape(-1, logits.shape[-1]),
                target_batch[1:, :].reshape(-1),
            )
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(list(self.dataset.train_loader))

    def evaluate(self):
        # Validate the model on the validaton dataset
        model.eval()
        for src_batch, target_batch in self.dataset.val_loader:
            src_batch = src_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            src_mask, target_mask = self.dataset.create_mask(src_batch, target_batch)

            output = self.model(src_batch, target_batch[:-1, :], src_mask, target_mask)
            loss = self.loss(
                output.reshape(-1, output.shape[-1]),
                target_batch[1:, :].reshape(-1),
            )
            print(loss.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    dataset = TranslationDataset(
        src_lang="en",
        tgt_lang="de",
        src_tokenizer=SpacyTokenizer(lang="en_core_web_sm"),
        tgt_tokenizer=SpacyTokenizer(lang="de_core_news_sm"),
    )

    model = Transformer(
        dim=512,
        vocab_size=dataset.get_vocab_size(type="src"),
    )

    trainer = Trainer(
        dataset,
        model,
    )

    for epoch in args.epochs:
        trainer.train_epoch()
        trainer.evaluate()
