"""Inference helpers.

``predict_image(path)`` is the simple API requested for deployment: it accepts
a file path (or an uploaded file object) and returns a :class:`Prediction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config.config import DATA_DIR, MODEL_PATH
from .data import decode_image_bytes, resolve_class_names
from .model import load_trained_model


@dataclass(frozen=True)
class Prediction:
    """Result of one image prediction."""

    label: str
    confidence: float
    probabilities: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (handy for APIs/UI)."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence:.2%} confidence)"


def _read_image_bytes(source: str | Path | Any) -> bytes:
    """Read bytes from a path or from a file-like object (Streamlit upload)."""
    if hasattr(source, "read"):
        data = source.read()
        try:
            source.seek(0)  # allow the caller/UI to display the upload again
        except (AttributeError, OSError):
            pass
    else:
        data = Path(source).read_bytes()
    return data


def preprocess_image(source: str | Path | Any) -> np.ndarray:
    """Return a (1, 224, 224, 3) float32 batch for one image."""
    image = decode_image_bytes(_read_image_bytes(source))
    return np.expand_dims(image.numpy(), axis=0)


def predict_image(
    path: str | Path | Any,
    model: Any | None = None,
    model_path: str | Path = MODEL_PATH,
    data_dir: str | Path = DATA_DIR,
    class_names: Sequence[str] | None = None,
) -> Prediction:
    """Classify one MRI image using the saved model (no training involved).

    Args:
        path: Image path, or any object with a ``read()`` method.
        model: Optional already-loaded Keras model (avoids reloading in a UI).
        model_path: Saved model used when ``model`` is not supplied.
        data_dir: Fallback source of class names for legacy models.
        class_names: Explicit class order; overrides metadata/directory scan.

    Returns:
        Prediction with the winning label, its confidence and all probabilities.
    """
    names = resolve_class_names(model_path, data_dir, class_names)
    if model is None:
        model = load_trained_model(model_path, compile_model=False)

    batch = preprocess_image(path)
    scores = np.asarray(model.predict(batch, verbose=0))[0].reshape(-1)

    if len(scores) != len(names):
        raise ValueError(
            f"Model has {len(scores)} outputs but {len(names)} class names were "
            "resolved. Check --model-path / --data-dir or the metadata sidecar."
        )

    index = int(np.argmax(scores))
    probabilities = {name: float(score) for name, score in zip(names, scores)}
    return Prediction(
        label=names[index],
        confidence=float(scores[index]),
        probabilities=probabilities,
    )


def predict_batch(
    paths: Sequence[str | Path | Any],
    model: Any | None = None,
    model_path: str | Path = MODEL_PATH,
    data_dir: str | Path = DATA_DIR,
    class_names: Sequence[str] | None = None,
) -> list[Prediction]:
    """Classify several images while loading the model only once."""
    names = resolve_class_names(model_path, data_dir, class_names)
    if model is None:
        model = load_trained_model(model_path, compile_model=False)
    return [
        predict_image(path, model=model, model_path=model_path, class_names=names)
        for path in paths
    ]
