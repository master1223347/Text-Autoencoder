"""Evaluation and reconstruction utilities for the text autoencoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from config.config import BATCH_SIZE, MAX_SEQUENCE_LENGTH, SAVED_DIR, VALIDATION_SPLIT
from src.dataloader import TextVocabulary, create_dataloaders
from src.model import TextAutoencoder
from src.utils import get_device


def load_trained_components(
    model_path: str | Path,
    device: torch.device,
) -> tuple[TextAutoencoder, TextVocabulary, dict[str, object]]:
    """Load a trained text autoencoder and its vocabulary from disk."""
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint["model_config"]
    vocabulary = TextVocabulary.from_dict(checkpoint["vocabulary"])

    model = TextAutoencoder(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
        pad_token_id=model_config["pad_token_id"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocabulary, checkpoint


def _prepare_single_text(
    text: str,
    vocabulary: TextVocabulary,
    max_sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one text string into model-ready tensors."""
    encoded = vocabulary.encode_text(text, max_sequence_length)
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)
    lengths = encoded["length"].unsqueeze(0).to(device)
    return input_ids, lengths


def encode_text(text: str, model_path: str | Path) -> dict[str, object]:
    """Encode one text string into a latent vector."""
    device = get_device()
    model, vocabulary, checkpoint = load_trained_components(model_path, device=device)
    max_sequence_length = checkpoint["model_config"].get("max_sequence_length", MAX_SEQUENCE_LENGTH)
    input_ids, lengths = _prepare_single_text(text, vocabulary, max_sequence_length, device)

    with torch.no_grad():
        latent = model.encode(input_ids, lengths)

    return {
        "input_text": text,
        "latent_vector": latent.squeeze(0).cpu().tolist(),
    }


def decode_latent(
    latent_vector: list[float],
    model_path: str | Path,
    max_length: int | None = None,
) -> dict[str, object]:
    """Decode a saved latent vector back into text."""
    device = get_device()
    model, vocabulary, checkpoint = load_trained_components(model_path, device=device)
    target_length = max_length or checkpoint["model_config"].get("max_sequence_length", MAX_SEQUENCE_LENGTH)
    latent = torch.tensor([latent_vector], dtype=torch.float32, device=device)

    with torch.no_grad():
        generated_ids = model.generate(latent, vocabulary.bos_id, vocabulary.eos_id, target_length)

    reconstruction = vocabulary.decode_ids(generated_ids[0].tolist())
    return {
        "latent_vector": latent_vector,
        "reconstruction": reconstruction,
    }


def reconstruct_text(text: str, model_path: str | Path) -> dict[str, object]:
    """Encode and immediately decode one text string."""
    device = get_device()
    model, vocabulary, checkpoint = load_trained_components(model_path, device=device)
    max_sequence_length = checkpoint["model_config"].get("max_sequence_length", MAX_SEQUENCE_LENGTH)
    input_ids, lengths = _prepare_single_text(text, vocabulary, max_sequence_length, device)

    with torch.no_grad():
        latent = model.encode(input_ids, lengths)
        generated_ids = model.generate(latent, vocabulary.bos_id, vocabulary.eos_id, max_sequence_length)

    reconstruction = vocabulary.decode_ids(generated_ids[0].tolist())
    return {
        "input_text": text,
        "reconstruction": reconstruction,
        "latent_vector": latent.squeeze(0).cpu().tolist(),
    }


def evaluate_autoencoder(
    data_path: str | Path,
    model_path: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
) -> dict[str, object]:
    """Evaluate a saved text autoencoder on validation text samples."""
    device = get_device()
    checkpoint_path = Path(model_path) if model_path is not None else SAVED_DIR / "autoencoder.pt"
    model, vocabulary, checkpoint = load_trained_components(checkpoint_path, device=device)
    max_sequence_length = checkpoint["model_config"].get("max_sequence_length", MAX_SEQUENCE_LENGTH)

    _, validation_loader, _ = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        validation_split=validation_split,
        max_sequence_length=max_sequence_length,
        vocabulary=vocabulary,
        shuffle=False,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id, reduction="sum")

    total_loss = 0.0
    total_tokens = 0
    sample_input = None
    sample_output = None
    sample_latent = None

    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch["input_ids"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            lengths = batch["length"].to(device)

            logits = model(input_ids, lengths, decoder_input_ids)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            latent = model.encode(input_ids, lengths)
            generated_ids = model.generate(latent, vocabulary.bos_id, vocabulary.eos_id, max_sequence_length)

            total_loss += loss.item()
            total_tokens += int(target_ids.ne(vocabulary.pad_id).sum().item())

            if sample_input is None:
                sample_input = batch["text"][0]
                sample_output = vocabulary.decode_ids(generated_ids[0].tolist())
                sample_latent = latent[0].cpu().tolist()

    average_loss = total_loss / total_tokens
    results = {
        "loss": average_loss,
        "sample_input_text": sample_input,
        "sample_reconstruction": sample_output,
        "sample_latent_vector": sample_latent,
    }

    print(f"Validation reconstruction loss: {average_loss:.6f}")
    return results
