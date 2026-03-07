"""Utilities for loading text samples into PyTorch dataloaders."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from config.config import (
    BATCH_SIZE,
    BOS_TOKEN,
    EOS_TOKEN,
    MAX_SEQUENCE_LENGTH,
    PAD_TOKEN,
    RANDOM_SEED,
    UNK_TOKEN,
    VALIDATION_SPLIT,
)


def load_text_samples(data_path: str | Path) -> list[str]:
    """Load non-empty text lines from disk."""
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    text = path.read_text(encoding="utf-8")
    samples = [line.strip() for line in text.splitlines() if line.strip()]

    if not samples and text.strip():
        samples = [text.strip()]
    if not samples:
        raise ValueError("No text samples were found in the dataset file")

    return samples


@dataclass
class TextVocabulary:
    """Character-level vocabulary used for encoding and decoding text."""

    token_to_id: dict[str, int]

    def __post_init__(self) -> None:
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}

    @classmethod
    def build(cls, samples: list[str]) -> "TextVocabulary":
        """Build a character vocabulary from training samples."""
        token_to_id = {
            PAD_TOKEN: 0,
            BOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        character_counts = Counter()
        for sample in samples:
            character_counts.update(sample)

        for character in sorted(character_counts):
            if character not in token_to_id:
                token_to_id[character] = len(token_to_id)

        return cls(token_to_id=token_to_id)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TextVocabulary":
        """Rebuild a vocabulary from serialized checkpoint data."""
        token_to_id = {str(token): int(token_id) for token, token_id in payload["token_to_id"].items()}
        return cls(token_to_id=token_to_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize the vocabulary for checkpoint storage."""
        return {"token_to_id": self.token_to_id}

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode_text(self, text: str, max_sequence_length: int) -> dict[str, torch.Tensor]:
        """Convert one text sample into encoder and decoder token tensors."""
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")

        content_limit = max_sequence_length - 1
        token_ids = [self.token_to_id.get(character, self.unk_id) for character in text[:content_limit]]

        encoder_ids = token_ids + [self.eos_id]
        decoder_input_ids = [self.bos_id] + token_ids
        target_ids = token_ids + [self.eos_id]

        sequence_length = len(encoder_ids)
        padding = [self.pad_id] * (max_sequence_length - sequence_length)

        return {
            "input_ids": torch.tensor(encoder_ids + padding, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids + padding, dtype=torch.long),
            "target_ids": torch.tensor(target_ids + padding, dtype=torch.long),
            "length": torch.tensor(sequence_length, dtype=torch.long),
        }

    def decode_ids(self, token_ids: list[int]) -> str:
        """Convert token ids back into a text string."""
        characters: list[str] = []
        for token_id in token_ids:
            if token_id in (self.pad_id, self.bos_id):
                continue
            if token_id == self.eos_id:
                break
            characters.append(self.id_to_token.get(token_id, ""))
        return "".join(characters)


class TextAutoencoderDataset(Dataset):
    """Dataset of text samples prepared for sequence-to-sequence reconstruction."""

    def __init__(self, samples: list[str], vocabulary: TextVocabulary, max_sequence_length: int) -> None:
        self.samples = samples
        self.vocabulary = vocabulary
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        encoded = self.vocabulary.encode_text(sample, self.max_sequence_length)
        encoded["text"] = sample
        return encoded


def create_dataloaders(
    data_path: str | Path,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    vocabulary: TextVocabulary | None = None,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, TextVocabulary]:
    """Create train and validation dataloaders for text autoencoder training."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be in the range [0, 1)")
    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least 2")

    samples = load_text_samples(data_path)
    if vocabulary is None:
        vocabulary = TextVocabulary.build(samples)

    dataset = TextAutoencoderDataset(samples, vocabulary, max_sequence_length)
    dataset_size = len(dataset)

    validation_size = int(dataset_size * validation_split)
    if validation_split > 0 and dataset_size > 1:
        validation_size = max(1, validation_size)
    if validation_size >= dataset_size:
        validation_size = dataset_size - 1

    if validation_size <= 0:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_loader, validation_loader, vocabulary

    train_size = dataset_size - validation_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, vocabulary
