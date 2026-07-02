"""Dataset loading and preprocessing for VoidGPT 1 120K."""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import CharTokenizer


class CharDataset(Dataset):
    """Character-level dataset with sliding window.

    Each sample is a (input, target) pair where target is input shifted by 1.
    """

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        return x, y


def load_text(path: str | Path) -> str:
    """Load text from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def encode_text(text: str, tokenizer: CharTokenizer) -> torch.Tensor:
    """Encode text to tensor of token IDs."""
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long)


def create_datasets(
    text_path: str | Path,
    tokenizer: CharTokenizer,
    seq_len: int = 128,
    train_frac: float = 0.9,
) -> tuple[CharDataset, CharDataset]:
    """Create train and validation datasets from a text file.

    Args:
        text_path: path to text file
        tokenizer: character tokenizer
        seq_len: sequence length (context window)
        train_frac: fraction of data for training

    Returns:
        (train_dataset, val_dataset)
    """
    text = load_text(text_path)
    data = encode_text(text, tokenizer)

    split = int(train_frac * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_ds = CharDataset(train_data, seq_len)
    val_ds = CharDataset(val_data, seq_len)

    return train_ds, val_ds


def create_dataloaders(
    train_ds: CharDataset,
    val_ds: CharDataset,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader
