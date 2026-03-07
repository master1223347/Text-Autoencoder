"""Command-line entry point for training and evaluating the autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path

from config.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, OUTPUTS_DIR, SAVED_DIR
from src.test import evaluate_autoencoder
from src.train import train_autoencoder
from src.utils import ensure_project_dirs, log_message, save_json


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for project actions."""
    parser = argparse.ArgumentParser(description="Autoencoder project runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the autoencoder")
    train_parser.add_argument("data_path", help="Path to a numeric .npy, .csv, or .txt dataset")
    train_parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Optimizer learning rate",
    )
    train_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation",
    )
    train_parser.add_argument(
        "--save-path",
        default=str(SAVED_DIR / "autoencoder.pt"),
        help="Path to save the trained model weights",
    )
    train_parser.add_argument(
        "--history-path",
        default=str(OUTPUTS_DIR / "train_history.json"),
        help="Path to save training loss history as JSON",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained autoencoder")
    eval_parser.add_argument("data_path", help="Path to a numeric .npy, .csv, or .txt dataset")
    eval_parser.add_argument(
        "--model-path",
        default=str(SAVED_DIR / "autoencoder.pt"),
        help="Path to trained model weights",
    )
    eval_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Evaluation batch size")
    eval_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation",
    )
    eval_parser.add_argument(
        "--results-path",
        default=str(OUTPUTS_DIR / "evaluation_results.json"),
        help="Path to save evaluation results as JSON",
    )

    return parser


def main() -> None:
    """Run the selected training or evaluation workflow."""
    parser = build_parser()
    args = parser.parse_args()
    ensure_project_dirs()

    if args.command == "train":
        _, history = train_autoencoder(
            data_path=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            save_path=args.save_path,
        )
        history_path = save_json(history, args.history_path)
        log_message(f"Training history saved to {history_path}")
        return

    results = evaluate_autoencoder(
        data_path=args.data_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )
    serializable_results = {
        "loss": results["loss"],
        "sample_input": results["sample_input"].tolist(),
        "sample_reconstruction": results["sample_reconstruction"].tolist(),
    }
    results_path = save_json(serializable_results, args.results_path)
    log_message(f"Evaluation results saved to {results_path}")


if __name__ == "__main__":
    main()
