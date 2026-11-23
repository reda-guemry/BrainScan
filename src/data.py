"""Data discovery, preprocessing and ``tf.data`` pipelines.

This module replaces notebook cells 1-10.  Importantly, decoding/resizing is
shared by training, evaluation and inference so all three see identical pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMG_SIZE,
    MODEL_PATH,
    SEED,
    TEST_SIZE,
    VAL_SIZE,
    VALID_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# Class names / metadata
# ---------------------------------------------------------------------------
def discover_classes(data_dir: str | Path = DATA_DIR) -> tuple[str, ...]:
    """Return class names sorted alphabetically (same order as the notebook)."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    classes = tuple(sorted(p.name for p in data_path.iterdir() if p.is_dir()))
    if not classes:
        raise ValueError(f"No class folders found inside {data_path}")
    return classes


def metadata_path_for(model_path: str | Path = MODEL_PATH) -> Path:
    """Return the sidecar JSON that stores the class order of a saved model."""
    path = Path(model_path)
    return path.with_name(f"{path.stem}_classes.json")


def save_class_metadata(
    model_path: str | Path,
    class_names: Sequence[str],
    img_size: tuple[int, int] = IMG_SIZE,
) -> Path:
    """Save the class order next to the model so predictions stay reproducible."""
    path = metadata_path_for(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"class_names": list(class_names), "img_size": list(img_size)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_class_metadata(model_path: str | Path = MODEL_PATH) -> dict | None:
    """Read the metadata sidecar if it exists (old models have none)."""
    path = metadata_path_for(model_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_class_names(
    model_path: str | Path = MODEL_PATH,
    data_dir: str | Path = DATA_DIR,
    class_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve class order from an explicit list, model metadata, or data dir."""
    if class_names:
        return tuple(class_names)

    metadata = load_class_metadata(model_path)
    if metadata and metadata.get("class_names"):
        return tuple(metadata["class_names"])

    # The existing models/best_model.keras was trained before metadata existed;
    # the alphabetically sorted data folders reproduce the notebook's order.
    return discover_classes(data_dir)


# ---------------------------------------------------------------------------
# File collection and splitting
# ---------------------------------------------------------------------------
def collect_samples(
    data_dir: str | Path = DATA_DIR,
    class_names: Sequence[str] | None = None,
) -> tuple[list[str], list[int], tuple[str, ...]]:
    """Collect image paths and integer labels: (paths, labels, class_names)."""
    names = tuple(class_names) if class_names else discover_classes(data_dir)
    class_to_id = {name: index for index, name in enumerate(names)}

    paths: list[str] = []
    labels: list[int] = []
    for class_name in names:
        class_dir = Path(data_dir) / class_name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                paths.append(str(image_path))
                labels.append(class_to_id[class_name])

    if not paths:
        raise ValueError(f"No images with extensions {VALID_EXTENSIONS} in {data_dir}")
    return paths, labels, names


def split_samples(
    paths: Sequence[str],
    labels: Sequence[int],
    seed: int = SEED,
) -> tuple[tuple[list[str], list[int]], ...]:
    """Stratified 70/15/15 split into (train, validation, test) tuples."""
    x_train, x_temp, y_train, y_temp = train_test_split(
        list(paths),
        list(labels),
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=list(labels),
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=VAL_SIZE,
        random_state=seed,
        stratify=y_temp,
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ---------------------------------------------------------------------------
# Image decoding / tf.data pipeline
# ---------------------------------------------------------------------------
def _decode_and_resize(image_bytes: tf.Tensor) -> tf.Tensor:
    """Decode JPEG/PNG/BMP bytes to RGB and resize to the model input size.

    ``tf.image.resize`` returns float32 pixels in the original 0..255 range.
    The notebook trained on exactly that scale, so we intentionally do NOT
    divide by 255 here (dividing here would corrupt predictions).
    """
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return image


def decode_image_bytes(data: bytes) -> tf.Tensor:
    """Decode raw image bytes (used for Streamlit uploads and ``predict_image``)."""
    if not data:
        raise ValueError("Empty image data")
    return _decode_and_resize(tf.constant(data, dtype=tf.string))


def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Map function used by ``Dataset.map``: file path -> preprocessed image."""
    image_bytes = tf.io.read_file(path)
    return _decode_and_resize(image_bytes), tf.cast(label, tf.int32)


def make_dataset(
    paths: Sequence[str],
    labels: Sequence[int],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Build a batched, prefetched dataset from image paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(list(paths)), tf.constant(list(labels), dtype=tf.int32))
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 10_000),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
