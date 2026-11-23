"""Evaluation of a saved model on the held-out test split.

This never trains; it only loads an existing model and runs
``evaluate``/``predict`` once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from .config.config import (
    BATCH_SIZE,
    CONFUSION_MATRIX_PATH,
    DATA_DIR,
    EVALUATION_PATH,
    MODEL_PATH,
)
from .data import collect_samples, make_dataset, resolve_class_names, split_samples
from .model import load_trained_model


def evaluate_model(
    model_path: str | Path = MODEL_PATH,
    data_dir: str | Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    max_samples: int | None = None,
    output_path: str | Path = EVALUATION_PATH,
    confusion_matrix_path: str | Path = CONFUSION_MATRIX_PATH,
    save_plot: bool = True,
) -> dict:
    """Evaluate on the test split and save metrics + confusion matrix."""
    class_names = resolve_class_names(model_path, data_dir)
    paths, labels, _ = collect_samples(data_dir, class_names)
    _, _, (x_test, y_test) = split_samples(paths, labels)

    if max_samples is not None:
        x_test, y_test = x_test[:max_samples], y_test[:max_samples]

    dataset = make_dataset(x_test, y_test, batch_size=batch_size)
    model = load_trained_model(model_path, compile_model=True)

    loss, accuracy = model.evaluate(dataset, verbose=0)
    probabilities = np.asarray(model.predict(dataset, verbose=0))
    y_pred = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(
        np.asarray(y_test, dtype=int), y_pred, labels=list(range(len(class_names)))
    )

    results = {
        "model_path": str(model_path),
        "test_samples": len(x_test),
        "loss": float(loss),
        "accuracy": float(accuracy),
        "class_names": list(class_names),
        "confusion_matrix": matrix.tolist(),
    }

    result_file = Path(output_path)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if save_plot:
        _save_confusion_matrix_plot(matrix, class_names, confusion_matrix_path)

    return results


def _save_confusion_matrix_plot(matrix, class_names, path: str | Path) -> None:
    """Write the confusion-matrix heatmap to disk (no GUI required)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, axes = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes,
    )
    axes.set_xlabel("Prédictions")
    axes.set_ylabel("Vraies classes")
    axes.set_title("Matrice de confusion")
    figure.tight_layout()
    plot_path = Path(path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
