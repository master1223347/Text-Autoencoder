# Autoencoder Project

This project is a PyTorch-based text autoencoder. It learns to encode text into a latent vector and decode that vector back into reconstructed text.

The current implementation includes:
- a character-level GRU encoder-decoder autoencoder
- text dataset loading from `.txt`
- vocabulary building from training text
- training with validation loss tracking
- text reconstruction from saved checkpoints
- latent vector encoding and decoding commands

## Project Structure

- `main.py`: Command-line entry point for training, evaluation, encoding, decoding, and reconstruction.
- `config/config.py`: Shared project settings and output paths.
- `data/`: Local datasets or sample data files.
- `src/dataloader.py`: Text loading, vocabulary building, and dataloader creation.
- `src/model.py`: Text autoencoder model definition.
- `src/train.py`: Training loop and checkpoint saving.
- `src/test.py`: Evaluation, checkpoint loading, and text reconstruction helpers.
- `src/utils.py`: Shared helpers for directories, devices, and JSON output.
- `saved/`: Saved model checkpoints.
- `outputs/`: Saved training history and evaluation results.

## Dataset Format

Training data should be a UTF-8 `.txt` file.
Each non-empty line is treated as one training sample.

Example:

```text
hello world
this is a text autoencoder
sequence models can reconstruct short sentences
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model on a text file:

```bash
python3 main.py train data/sampledata.txt
```

Evaluate reconstruction loss on the dataset:

```bash
python3 main.py evaluate data/sampledata.txt
```

Reconstruct one string with a trained model:

```bash
python3 main.py reconstruct "hello world"
```

Encode one string into a latent vector:

```bash
python3 main.py encode "hello world"
```

Decode a saved latent vector back into text:

```bash
python3 main.py decode outputs/encoded_text.json
```

## Outputs

Training saves:
- a model checkpoint in `saved/`
- a JSON loss history in `outputs/`

Evaluation saves:
- reconstruction loss
- one sample input text
- one sample reconstruction
- one sample latent vector

Reconstruction and encoding commands also save JSON outputs in `outputs/`.

## Important Note

This project learns a latent representation of text, but that does not automatically guarantee better real-world file compression than standard compression tools. The encoded latent vector is the model's internal representation for reconstruction.
