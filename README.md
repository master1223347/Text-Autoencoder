# Autoencoder Project

This project is a simple PyTorch-based autoencoder workspace for learning, training, and evaluating reconstruction models on numeric data.

The current implementation includes:
- a fully connected autoencoder model
- numeric data loading from `.npy`, `.csv`, and `.txt`
- training with validation loss tracking
- evaluation using saved checkpoints
- JSON output files for training history and evaluation results

## Project Structure

- `main.py`: Command-line entry point for training and evaluation.
- `config/config.py`: Shared project settings and output paths.
- `data/`: Local datasets or sample data files.
- `src/dataloader.py`: Data loading and PyTorch dataloader creation.
- `src/model.py`: Autoencoder model definition.
- `src/train.py`: Training loop and checkpoint saving.
- `src/test.py`: Evaluation and checkpoint loading.
- `src/utils.py`: Shared helpers for directories, devices, and JSON output.
- `saved/`: Saved model checkpoints.
- `outputs/`: Saved training history and evaluation results.

## Supported Data Formats

The dataloader currently supports numeric datasets stored as:
- `.npy`
- `.csv`
- `.txt`

Each row is treated as one sample. For autoencoder training, the same sample is used as both the input and reconstruction target.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python3 main.py train path/to/data.csv
```

Evaluate a trained model:

```bash
python3 main.py evaluate path/to/data.csv
```

You can also override defaults such as `--epochs`, `--batch-size`, `--learning-rate`, `--save-path`, and `--results-path`.

## Outputs

Training saves:
- a model checkpoint in `saved/`
- a JSON loss history in `outputs/`

Evaluation saves:
- reconstruction loss
- one sample input
- one sample reconstruction

These evaluation outputs are written as JSON in `outputs/`.
