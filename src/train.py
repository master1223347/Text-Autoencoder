"""Training utilities for the autoencoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config.config import BATCH_SIZE, EPOCHS, HIDDEN_DIM, LATENT_DIM, LEARNING_RATE, SAVED_DIR
from src.dataloader import create_dataloaders
from src.model import Autoencoder
from src.utils import ensure_project_dirs, get_device


def _run_epoch(
    model: Autoencoder,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> float:
    """Run one training or evaluation epoch and return the average loss."""
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets.view(targets.size(0), -1))

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train_autoencoder(
    data_path: str | Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    validation_split: float = 0.2,
    save_path: str | Path | None = None,
) -> tuple[Autoencoder, dict[str, list[float]]]:
    """Train an autoencoder on numeric data and save the learned weights."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")

    train_loader, validation_loader = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        validation_split=validation_split,
    )

    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.view(sample_batch.size(0), -1).size(1)

    device = get_device()
    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer)
        val_loss = _run_epoch(model, validation_loader, loss_fn, device)

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
    torch.save(model.state_dict(), target_path)

    return model, history
