"""Command-line entry point for the text autoencoder."""

from __future__ import annotations

import argparse

from config.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUTS_DIR,
    SAVED_DIR,
    VALIDATION_SPLIT,
)
from src.test import decode_latent, encode_text, evaluate_autoencoder, reconstruct_text
from src.train import train_autoencoder
from src.utils import ensure_project_dirs, load_json, log_message, save_json


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for project actions."""
    parser = argparse.ArgumentParser(description="Text autoencoder runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the autoencoder")
    train_parser.add_argument("data_path", help="Path to a text dataset file")
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
        default=VALIDATION_SPLIT,
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
    eval_parser.add_argument("data_path", help="Path to a text dataset file")
    eval_parser.add_argument(
        "--model-path",
        default=str(SAVED_DIR / "autoencoder.pt"),
        help="Path to trained model weights",
    )
    eval_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Evaluation batch size")
    eval_parser.add_argument(
        "--validation-split",
        type=float,
        default=VALIDATION_SPLIT,
        help="Fraction of data reserved for validation",
    )
    eval_parser.add_argument(
        "--results-path",
        default=str(OUTPUTS_DIR / "evaluation_results.json"),
        help="Path to save evaluation results as JSON",
    )

    reconstruct_parser = subparsers.add_parser("reconstruct", help="Encode and reconstruct one text string")
    reconstruct_parser.add_argument("text", help="Text string to reconstruct")
    reconstruct_parser.add_argument(
        "--model-path",
        default=str(SAVED_DIR / "autoencoder.pt"),
        help="Path to trained model weights",
    )
    reconstruct_parser.add_argument(
        "--results-path",
        default=str(OUTPUTS_DIR / "reconstruction.json"),
        help="Path to save reconstruction output as JSON",
    )

    encode_parser = subparsers.add_parser("encode", help="Encode one text string into a latent vector")
    encode_parser.add_argument("text", help="Text string to encode")
    encode_parser.add_argument(
        "--model-path",
        default=str(SAVED_DIR / "autoencoder.pt"),
        help="Path to trained model weights",
    )
    encode_parser.add_argument(
        "--output-path",
        default=str(OUTPUTS_DIR / "encoded_text.json"),
        help="Path to save the latent vector as JSON",
    )

    decode_parser = subparsers.add_parser("decode", help="Decode a saved latent vector back into text")
    decode_parser.add_argument("latent_path", help="Path to a JSON file containing a latent_vector field")
    decode_parser.add_argument(
        "--model-path",
        default=str(SAVED_DIR / "autoencoder.pt"),
        help="Path to trained model weights",
    )
    decode_parser.add_argument(
        "--results-path",
        default=str(OUTPUTS_DIR / "decoded_text.json"),
        help="Path to save decoded text as JSON",
    )

    return parser


def main() -> None:
    """Run the selected training or evaluation workflow."""
    parser = build_parser()
    args = parser.parse_args()
    ensure_project_dirs()

    if args.command == "train":
        _, _, history = train_autoencoder(
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

    if args.command == "evaluate":
        results = evaluate_autoencoder(
            data_path=args.data_path,
            model_path=args.model_path,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
        )
        results_path = save_json(results, args.results_path)
        log_message(f"Evaluation results saved to {results_path}")
        return

    if args.command == "reconstruct":
        results = reconstruct_text(args.text, args.model_path)
        results_path = save_json(results, args.results_path)
        log_message(f"Reconstructed text: {results['reconstruction']}")
        log_message(f"Reconstruction results saved to {results_path}")
        return

    if args.command == "encode":
        results = encode_text(args.text, args.model_path)
        output_path = save_json(results, args.output_path)
        log_message(f"Encoded latent vector saved to {output_path}")
        return

    latent_payload = load_json(args.latent_path)
    results = decode_latent(latent_payload["latent_vector"], args.model_path)
    results_path = save_json(results, args.results_path)
    log_message(f"Decoded text: {results['reconstruction']}")
    log_message(f"Decoded text saved to {results_path}")


if __name__ == "__main__":
    main()
