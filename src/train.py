"""Training utilities for the text autoencoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config.config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    EPOCHS,
    HIDDEN_DIM,
    LATENT_DIM,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    SAVED_DIR,
    VALIDATION_SPLIT,
)
from src.dataloader import TextVocabulary, create_dataloaders
from src.model import TextAutoencoder
from src.utils import ensure_project_dirs, get_device


def _run_epoch(
    model: TextAutoencoder,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    pad_token_id: int,
    device: torch.device,
    optimizer: Adam | None = None,
) -> float:
    """Run one training or evaluation epoch and return average token loss."""
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        lengths = batch["length"].to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(input_ids, lengths, decoder_input_ids)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        token_count = int(target_ids.ne(pad_token_id).sum().item())
        total_loss += loss.item()
        total_tokens += token_count

    return total_loss / total_tokens


def train_autoencoder(
    data_path: str | Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    validation_split: float = VALIDATION_SPLIT,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    save_path: str | Path | None = None,
) -> tuple[TextAutoencoder, TextVocabulary, dict[str, list[float]]]:
    """Train a text autoencoder and save a checkpoint with vocabulary data."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")

    train_loader, validation_loader, vocabulary = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        validation_split=validation_split,
        max_sequence_length=max_sequence_length,
    )

    device = get_device()
    model = TextAutoencoder(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        pad_token_id=vocabulary.pad_id,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id, reduction="sum")

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss = _run_epoch(
            model,
            train_loader,
            loss_fn,
            vocabulary.pad_id,
            device,
            optimizer=optimizer,
        )
        val_loss = _run_epoch(
            model,
            validation_loader,
            loss_fn,
            vocabulary.pad_id,
            device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- train_loss: {train_loss:.6f} "
            f"- val_loss: {val_loss:.6f}"
        )

    ensure_project_dirs()
    target_path = Path(save_path) if save_path is not None else SAVED_DIR / "autoencoder.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": len(vocabulary),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "pad_token_id": vocabulary.pad_id,
            "max_sequence_length": max_sequence_length,
        },
        "vocabulary": vocabulary.to_dict(),
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_split": validation_split,
        },
        "history": history,
    }
    torch.save(checkpoint, target_path)

    return model, vocabulary, history
