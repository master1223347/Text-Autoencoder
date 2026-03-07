"""Evaluation utilities for the autoencoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from config.config import BATCH_SIZE, HIDDEN_DIM, LATENT_DIM, SAVED_DIR
from src.dataloader import create_dataloaders
from src.model import Autoencoder
from src.utils import get_device


def load_trained_model(
    model_path: str | Path,
    device: torch.device,
) -> Autoencoder:
    """Load a trained autoencoder from disk."""
    checkpoint = torch.load(model_path, map_location=device)

    if "state_dict" in checkpoint:
        model_config = checkpoint.get("model_config", {})
        input_dim = model_config["input_dim"]
        hidden_dim = model_config.get("hidden_dim", HIDDEN_DIM)
        latent_dim = model_config.get("latent_dim", LATENT_DIM)
        state_dict = checkpoint["state_dict"]
    else:
        # Support older checkpoints that only stored the model weights.
        state_dict = checkpoint
        encoder_weight = state_dict["encoder.0.weight"]
        decoder_hidden_weight = state_dict["decoder.0.weight"]
        input_dim = encoder_weight.shape[1]
        hidden_dim = encoder_weight.shape[0]
        latent_dim = decoder_hidden_weight.shape[1]

    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)
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

    device = get_device()
    checkpoint_path = Path(model_path) if model_path is not None else SAVED_DIR / "autoencoder.pt"
    model = load_trained_model(checkpoint_path, device=device)
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
