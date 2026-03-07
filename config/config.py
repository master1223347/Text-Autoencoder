"""Project configuration values used across training and evaluation."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAVED_DIR = PROJECT_ROOT / "saved"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Core training settings.
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# Model settings for a simple fully connected autoencoder.
INPUT_DIM = 784
HIDDEN_DIM = 128
LATENT_DIM = 16
