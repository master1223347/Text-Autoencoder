"""Shared utility helpers for training and evaluation workflows."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from config.config import OUTPUTS_DIR, SAVED_DIR


def ensure_project_dirs() -> None:
    """Create output directories used by training and evaluation."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Return the best available device for PyTorch."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(data: dict, output_path: str | Path) -> Path:
    """Save a dictionary to a JSON file and return the file path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def log_message(message: str) -> None:
    """Print a project message."""
    print(message)
