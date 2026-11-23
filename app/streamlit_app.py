"""Streamlit UI for testing predictions with a saved BrainScan model.

Run from the repository root:

    uv run streamlit run app/streamlit_app.py

The interface never trains.  If the model file is missing it tells the user to
run ``python -m src.main --train True`` from a terminal instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Allow `streamlit run app/streamlit_app.py` from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.config import DATA_DIR, MODEL_PATH, VALID_EXTENSIONS  # noqa: E402
from src.data import resolve_class_names  # noqa: E402
from src.model import load_trained_model  # noqa: E402
from src.predict import predict_image  # noqa: E402

st.set_page_config(page_title="BrainScan", page_icon="🧠", layout="wide")
st.title("🧠 BrainScan — MRI tumor classification")
st.markdown(
    "Upload a brain MRI image (or pick a sample) to test the **saved** Keras model. "
    "Training is not available from this page — use "
    "`python -m src.main --train True` in a terminal for that."
)


@st.cache_resource(show_spinner="Loading Keras model...")
def cached_model(model_path: str, modified_time: float):
    """Cache the loaded model; ``modified_time`` invalidates it after retraining."""
    return load_trained_model(model_path)


def sample_images(limit_per_class: int = 8) -> list[Path]:
    """Collect a few example images so the app is usable without an upload."""
    if not DATA_DIR.is_dir():
        return []
    samples: list[Path] = []
    for class_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        files = sorted(
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        samples.extend(files[:limit_per_class])
    return samples


# -------------------------------------------------------------------------
# Sidebar: model selection and status
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    model_path_text = st.text_input("Saved model path", str(MODEL_PATH))
    model_path = Path(model_path_text)

    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.info("Run `python -m src.main --train True` to create it, then refresh.")
        st.stop()

    modified_time = model_path.stat().st_mtime
    st.success(f"Loaded from {model_path.name}")
    st.caption(f"{model_path.stat().st_size / 1e6:.1f} MB · modified {modified_time:.0f}")
    try:
        class_names = resolve_class_names(model_path, DATA_DIR)
        st.write("Classes:", ", ".join(class_names))
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

st.markdown("---")

# -------------------------------------------------------------------------
# Image input: upload or sample
# -------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload an MRI image",
    type=[ext.lstrip(".") for ext in VALID_EXTENSIONS],
)
samples = sample_images()
sample_choice = st.selectbox(
    "…or choose a sample from data/",
    options=[None, *samples],
    format_func=lambda value: "No sample selected" if value is None else str(value),
)

source = uploaded if uploaded is not None else sample_choice
left, right = st.columns([1, 1])

with left:
    if source is None:
        st.info("Choose an image to get started.")
    else:
        try:
            with Image.open(source) as image:
                st.image(image.convert("RGB"), caption="Input image", use_container_width=True)
        except Exception as error:  # Streamlit displays the problem to the user
            st.error(f"Could not open image: {error}")
            source = None

with right:
    st.subheader("Prediction")
    if st.button("🔍 Predict", type="primary", disabled=source is None):
        try:
            model = cached_model(str(model_path), modified_time)
            result = predict_image(
                source,
                model=model,
                model_path=model_path,
                class_names=class_names,
            )
            st.metric("Class", result.label, f"{result.confidence:.2%} confidence")
            probabilities = pd.DataFrame(
                {"Probability": result.probabilities}
            )
            st.bar_chart(probabilities)
            st.dataframe(
                probabilities.style.format("{:.2%}"),
                use_container_width=True,
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.markdown("---")
st.caption(
    "Research/demo only. This model is not a medical device and must not be "
    "used for diagnosis."
)
