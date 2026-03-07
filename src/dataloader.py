"""Utilities for loading numeric data into PyTorch dataloaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

from config.config import BATCH_SIZE


def load_data(data_path: str | Path) -> Tensor:
    """Load numeric samples from a .npy, .csv, or .txt file."""
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".csv":
        data = np.loadtxt(path, delimiter=",")
    elif path.suffix == ".txt":
        data = np.loadtxt(path)
    else:
        raise ValueError("Unsupported file format. Use .npy, .csv, or .txt")

    array = np.asarray(data, dtype=np.float32)

    # Keep each row as one sample and flatten higher-dimensional inputs.
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)

    return torch.from_numpy(array)


def build_autoencoder_dataset(data_path: str | Path) -> TensorDataset:
    """Return a dataset where each sample is both the input and target."""
    samples = load_data(data_path)
    return TensorDataset(samples, samples)


def create_dataloaders(
    data_path: str | Path,
    batch_size: int = BATCH_SIZE,
    validation_split: float = 0.2,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for autoencoder training."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be in the range [0, 1)")

    dataset = build_autoencoder_dataset(data_path)
    dataset_size = len(dataset)

    if dataset_size == 0:
        raise ValueError("Dataset is empty")

    validation_size = int(dataset_size * validation_split)
    if validation_split > 0 and dataset_size > 1:
        validation_size = max(1, validation_size)
    if validation_size >= dataset_size:
        validation_size = dataset_size - 1

    train_size = dataset_size - validation_size

    if validation_size == 0:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_loader, validation_loader

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader
