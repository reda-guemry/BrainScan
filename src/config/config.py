"""Central configuration for the BrainScan project.

All paths are derived from this file's location so the code works no matter
which directory the command is launched from.  The original constants used by
``notebooks/01_data_loading.ipynb`` are kept for backwards compatibility.
"""

from pathlib import Path

# config.py lives in <repo>/src/config/, so parents[2] is the repository root.
ROOT_DIR = Path(__file__).resolve().parents[2]

# Where the MRI images live: <repo>/data/<class_name>/*.jpg
ROOT = ROOT_DIR / "data"
DATA_DIR = ROOT

# Where trained models and run artifacts are written.
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.keras"
OUTPUTS_DIR = ROOT_DIR / "outputs"
HISTORY_PATH = MODELS_DIR / "training_history.json"
EVALUATION_PATH = MODELS_DIR / "evaluation.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"

# Image preprocessing (must match the notebook and the saved model: no /255
# rescaling is applied because the notebook trained on float32 pixels 0..255).
VALID_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp")
IMG_SIZE = (224, 224)
SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# 70% train / 15% validation / 15% test, exactly like the notebook's two splits.
TEST_SIZE = 0.30
VAL_SIZE = 0.50  # fraction of the temporary (val + test) set
