from typing import List, Iterable, Any
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
SPECIAL_TOKENS = ['<unk>', '<pad>', '<bos>', '<eos>']

class WordTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}

    def tokenize(self, text: str) -> List[int]:
        tokens = text.split()
        return [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

class CharTokenizer:
    def __init__(self):
        self.vocab = {chr(i): i for i in range(128)}
        self.vocab["<PAD>"] = 0

    def tokenize(self, text: str) -> List[int]:
        return [self.vocab.get(char, self.vocab["<PAD>"]) for char in text]


class SpacyTokenizer:
    def __init__(self, lang: str = "en_core_web_sm"):
        self.tokenizer = get_tokenizer("spacy", language=lang)
        self.vocab = None

    def build_vocab(self, data_iter: Iterable[Any], index: int = 0) -> None:
        def yield_tokens(dataset_iter: Iterable[Any]) -> Iterable[str]:
            for sample in dataset_iter:
                yield self.tokenizer(sample[index])
        self.vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=SPECIAL_TOKENS)
        self.vocab.set_default_index(UNK_IDX)
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)
    
    def tokenize(self, text: str, torch: bool = False) -> List[int]:
        token_ids = self.vocab(self.tokenizer(text))
        if torch:
            return torch.cat(
                (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
            )
        return token_ids