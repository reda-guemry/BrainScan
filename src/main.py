"""Command-line entry point.

Examples (from the repository root):

    python -m src.main                         # no training, just show status
    python -m src.main --predict data/glioma/Te-gl_0010.jpg
    python -m src.main --evaluate --max-eval-samples 64
    python -m src.main --train True --epochs 5  # retrain (calls model.fit)

Training is opt-in: passing ``--train False`` (or omitting the flag) never
calls ``model.fit`` and always reuses the saved model when it exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config.config import BATCH_SIZE, DATA_DIR, EPOCHS, MODEL_PATH
from .data import resolve_class_names
from .evaluate import evaluate_model
from .model import load_trained_model, model_summary_text
from .predict import predict_image
from .train import train_model


def parse_bool(value: str | bool) -> bool:
    """Argparse converter accepting True/False, 1/0, yes/no."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got '{value}'")


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI options."""
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="BrainScan: classify brain MRI images with a saved Keras model.",
    )
    parser.add_argument(
        "--train",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        metavar="BOOL",
        help="Retrain and overwrite the model (True/False, default: False).",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs when training.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--predict", metavar="IMAGE", help="Classify one image and exit.")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate on the held-out test split."
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=None, help="Optional evaluation subset."
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip confusion matrix PNG.")
    parser.add_argument("--list-classes", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print the Keras summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested action(s). Returns a process exit code."""
    args = build_parser().parse_args(argv)
    model_path = Path(args.model_path)

    try:
        if args.train:
            train_model(
                data_dir=args.data_dir,
                model_path=model_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

        if not model_path.exists():
            print(
                f"ERROR: no saved model at {model_path}.\n"
                "Nothing was trained because training is opt-in. "
                "Run: python -m src.main --train True"
            )
            return 2

        print(f"Using saved model: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
        class_names = resolve_class_names(model_path, args.data_dir)
        print(f"Classes: {dict(enumerate(class_names))}")

        if args.list_classes:
            for index, name in enumerate(class_names):
                print(f"  {index}: {name}")

        if args.summary:
            print(model_summary_text(load_trained_model(model_path)))

        if args.evaluate:
            results = evaluate_model(
                model_path=model_path,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                max_samples=args.max_eval_samples,
                save_plot=not args.no_plot,
            )
            print(
                f"Test accuracy: {results['accuracy']:.4f} | "
                f"loss: {results['loss']:.4f} | samples: {results['test_samples']}"
            )

        if args.predict:
            prediction = predict_image(
                args.predict, model_path=model_path, data_dir=args.data_dir
            )
            print(f"Prediction: {prediction}")
            for name, probability in prediction.probabilities.items():
                print(f"  {name:<12} {probability:6.2%}")

        if not (args.train or args.list_classes or args.summary or args.evaluate or args.predict):
            print(
                "Nothing else to do. Try --predict IMAGE, --evaluate, "
                "or --train True to retrain."
            )
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
