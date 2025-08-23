"""
Backward compatibility imports for the refactored ML Tokamak project.

DEPRECATED: These imports are provided for backward compatibility only.
Please update your code to use the new modular structure:

- from src.training.LHmode_classifier import train_and_test_ris_model
- from src.training.alt_models_training import train_and_test_alt_model
- from src.utils.confinement_mode_classifier import *
- etc.
"""

import warnings
warnings.warn(
    "Using compatibility imports. Please update to use the new src/ structure.",
    DeprecationWarning,
    stacklevel=2
)

# Provide backward compatibility imports
try:
    from src.training.LHmode_classifier import *
    from src.training.alt_models_training import *
    from src.training.PhyDNet_COMPASS import *
    from src.utils.confinement_mode_classifier import *
    from src.models.alt_models import *
    from src.models.PhyDNet_models import *
    from src.data_processing.imgs_processing import *
    from analysis.visual import *
except ImportError as e:
    print(f"Warning: Could not import from new structure: {e}")
    print("Please ensure all dependencies are installed and paths are correct.")
