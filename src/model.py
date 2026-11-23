"""Model definition, loading and saving helpers.

The architecture is the same as the one used in the original notebook, so the
existing ``models/best_model.keras`` file can be loaded directly.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from .config.config import IMG_SIZE, LEARNING_RATE, MODEL_PATH

SUPPORTED_MODEL_SUFFIXES = (".keras", ".h5")


def build_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (*IMG_SIZE, 3),
    learning_rate: float = LEARNING_RATE,
) -> Sequential:
    """Build and compile the small CNN from the notebook.

    Called only when training is explicitly requested (``--train True``).
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPool2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPool2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ],
        name="brainscan_cnn",
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


@lru_cache(maxsize=8)
def load_trained_model(model_path: str | Path = MODEL_PATH, compile_model: bool = False):
    """Load a saved model once and cache it (Streamlit reruns stay fast).

    ``compile_model=False`` is enough for prediction.  Evaluation needs the
    compiled loss/metrics, so pass ``compile_model=True`` there.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run `python -m src.main --train True` first."
        )
    if path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
        raise ValueError(
            f"Unsupported model extension '{path.suffix}'. "
            f"Use one of {SUPPORTED_MODEL_SUFFIXES}."
        )
    return load_model(str(path), compile=compile_model)


def save_model(model, model_path: str | Path = MODEL_PATH) -> Path:
    """Save a trained model as .keras (native) or legacy .h5."""
    path = Path(model_path)
    if path.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
        raise ValueError(
            f"Unsupported model extension '{path.suffix}'. "
            f"Use one of {SUPPORTED_MODEL_SUFFIXES}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path


def model_summary_text(model) -> str:
    """Return ``model.summary()`` as a string for CLI / UI display."""
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(f"{line}\n"))
    return buffer.getvalue()
