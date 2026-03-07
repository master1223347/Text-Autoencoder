"""Project configuration values for the text autoencoder."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAVED_DIR = PROJECT_ROOT / "saved"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Core training settings for text reconstruction.
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 20
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Text autoencoder model settings.
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
LATENT_DIM = 64
MAX_SEQUENCE_LENGTH = 120

# Special tokens used by the character-level vocabulary.
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
