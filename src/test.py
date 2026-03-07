"""Evaluation utilities for the autoencoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from config.config import BATCH_SIZE, HIDDEN_DIM, LATENT_DIM, SAVED_DIR
from src.dataloader import create_dataloaders
from src.model import Autoencoder


def load_trained_model(
    model_path: str | Path,
    input_dim: int,
    device: torch.device,
) -> Autoencoder:
    """Load a trained autoencoder from disk."""
    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_autoencoder(
    data_path: str | Path,
    model_path: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
    validation_split: float = 0.2,
) -> dict[str, object]:
    """Evaluate a saved autoencoder and return loss plus sample reconstructions."""
    _, validation_loader = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False,
    )

    sample_batch, _ = next(iter(validation_loader))
    input_dim = sample_batch.view(sample_batch.size(0), -1).size(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(model_path) if model_path is not None else SAVED_DIR / "autoencoder.pt"
    model = load_trained_model(checkpoint_path, input_dim=input_dim, device=device)
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_samples = 0
    sample_input = None
    sample_output = None

    with torch.no_grad():
        for inputs, targets in validation_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets.view(targets.size(0), -1))

            batch_size_value = inputs.size(0)
            total_loss += loss.item() * batch_size_value
            total_samples += batch_size_value

            if sample_input is None:
                sample_input = targets[0].view(-1).cpu()
                sample_output = outputs[0].view(-1).cpu()

    average_loss = total_loss / total_samples
    results = {
        "loss": average_loss,
        "sample_input": sample_input,
        "sample_reconstruction": sample_output,
    }

    print(f"Validation reconstruction loss: {average_loss:.6f}")
    return results
