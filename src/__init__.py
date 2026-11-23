"""BrainScan: reusable modules for brain MRI classification.

The original research code lived in ``notebooks/01_data_loading.ipynb``.  Each
notebook step now has a named function in this package so it can be imported,
tested and run from the command line without re-running training.
"""

import os

# Keep TensorFlow/Keras start-up messages quiet before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = "0.1.0"
