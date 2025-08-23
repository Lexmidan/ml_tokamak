#!/usr/bin/env python3
"""
Cross-validation script for COMPASS Tokamak models.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.cross_validation_resnet34 import run_cross_validation

if __name__ == "__main__":
    # Run cross-validation with default parameters
    results = run_cross_validation(
        model_name='resnet34',
        ris_option='RIS1',
        n_splits=5
    )
    print("Cross-validation completed!")
    print(f"Results: {results}")
