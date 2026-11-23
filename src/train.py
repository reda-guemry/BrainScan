"""Training entry point.

Nothing in this module trains anything on import: ``train_model()`` is called
only by ``src.main`` when the user passes ``--train True``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint

from .config.config import (
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    HISTORY_PATH,
    IMG_SIZE,
    MODEL_PATH,
    SEED,
)
from .data import collect_samples, make_dataset, save_class_metadata, split_samples
from .model import build_model, save_model


def train_model(
    data_dir: str | Path = DATA_DIR,
    model_path: str | Path = MODEL_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    history_path: str | Path = HISTORY_PATH,
) -> dict:
    """Retrain the CNN, overwrite the saved model, and write run artifacts.

    Returns a small summary dictionary (counts, duration, history).
    """
    # 1) Discover images/classes and make the stratified 70/15/15 split.
    paths, labels, class_names = collect_samples(data_dir)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_samples(
        paths, labels, seed=seed
    )
    print(f"Classes: {dict(enumerate(class_names))}")
    print(
        f"Images: {len(paths)} total | "
        f"{len(x_train)} train / {len(x_val)} val / {len(x_test)} test"
    )

    # 2) Build a fresh model (the notebook's architecture).
    model = build_model(num_classes=len(class_names))

    # 3) Keep the best epoch by validation accuracy.  The notebook monitored
    #    "val_recall", which never existed because only accuracy was compiled.
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    # 4) Train.  This is the ONLY call to model.fit() in the whole project.
    start = time.perf_counter()
    history = model.fit(
        make_dataset(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed),
        validation_data=make_dataset(x_val, y_val, batch_size=batch_size),
        epochs=epochs,
        callbacks=[checkpoint],
    )
    duration = time.perf_counter() - start

    # 5) Guarantee a saved model even if the checkpoint callback never fired.
    if not Path(model_path).exists():
        save_model(model, model_path)

    # 6) Persist class order + metrics for reproducibility.
    save_class_metadata(model_path, class_names, img_size=IMG_SIZE)
    summary = {
        "model_path": str(model_path),
        "class_names": list(class_names),
        "epochs": epochs,
        "duration_seconds": round(duration, 2),
        "split_sizes": {
            "train": len(x_train),
            "val": len(x_val),
            "test": len(x_test),
        },
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }
    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Training finished in {duration:.2f}s. Model saved to {model_path}")
    print(f"Metrics written to {history_file}")
    return summary
