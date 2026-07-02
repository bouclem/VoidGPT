"""Character-level tokenizer for VoidGPT 1 120K."""

import json
from pathlib import Path


SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}

NUM_SPECIAL = len(SPECIAL_TOKENS)


class CharTokenizer:
    """Character-level tokenizer with special tokens.

    Vocab layout: [special tokens] + [printable ASCII chars sorted by code].
    Default vocab size ~100 (4 special + 95 printable ASCII).
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        if vocab is not None:
            self.stoi = dict(vocab)
            self.itos = {i: s for s, i in self.stoi.items()}
            self.vocab_size = len(self.stoi)
        else:
            self._build_default_vocab()

    def _build_default_vocab(self):
        """Build vocab from special tokens + printable ASCII (0x20–0x7E)."""
        self.stoi = dict(SPECIAL_TOKENS)
        idx = NUM_SPECIAL
        for code in range(0x20, 0x7F):  # space through tilde
            ch = chr(code)
            self.stoi[ch] = idx
            idx += 1
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def build_from_text(self, text: str):
        """Build vocab from special tokens + all unique chars in text."""
        self.stoi = dict(SPECIAL_TOKENS)
        idx = NUM_SPECIAL
        for ch in sorted(set(text)):
            if ch not in self.stoi:
                self.stoi[ch] = idx
                idx += 1
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to list of token IDs."""
        ids = []
        if add_bos:
            ids.append(SPECIAL_TOKENS["<BOS>"])
        for ch in text:
            ids.append(self.stoi.get(ch, SPECIAL_TOKENS["<UNK>"]))
        if add_eos:
            ids.append(SPECIAL_TOKENS["<EOS>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode list of token IDs to string."""
        chars = []
        for i in ids:
            token = self.itos.get(i, "<UNK>")
            if token not in ("<PAD>", "<BOS>", "<EOS>", "<UNK>"):
                chars.append(token)
        return "".join(chars)

    def save(self, path: str | Path):
        """Save tokenizer vocab to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["stoi"])

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKENS["<PAD>"]

    @property
    def bos_id(self) -> int:
        return SPECIAL_TOKENS["<BOS>"]

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKENS["<EOS>"]

    @property
    def unk_id(self) -> int:
        return SPECIAL_TOKENS["<UNK>"]
