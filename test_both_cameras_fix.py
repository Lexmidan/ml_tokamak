#!/usr/bin/env python3
"""
Test script to verify the 'both' cameras fix works correctly.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import logging

# Add current directory to path so we can import the modules
sys.path.append(str(Path(__file__).parent))

from cross_validation_resnet34 import load_shot_data_for_both_cameras, create_dataloaders_for_both_cameras

def test_both_cameras_fix():
    """Test the new both cameras functionality."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    path = Path(os.getcwd())
    logger.info(f"Testing both cameras fix in {path}")
    
    # Test 1: Load shot data for both cameras
    logger.info("Test 1: Loading shot data for both cameras...")
    try:
        shots_for_testing, shots_for_validation, shots_for_training = load_shot_data_for_both_cameras(
            path, test_df_contains_val_df=False, test_run=True, data_frac=1.0, random_seed=42
        )
        
        logger.info(f"✓ Shot data loaded successfully:")
        logger.info(f"  Training shots: {len(shots_for_training)}")
        logger.info(f"  Validation shots: {len(shots_for_validation)}")  
        logger.info(f"  Testing shots: {len(shots_for_testing)}")
        
        # Print some example shots
        logger.info(f"  Example training shots: {shots_for_training.head().tolist()}")
        
    except Exception as e:
        logger.error(f"✗ Failed to load shot data: {e}")
        return False
    
    # Test 2: Verify shot usage file exists and has expected structure
    logger.info("Test 2: Checking shot usage file...")
    try:
        shot_usage = pd.read_csv(f'{path}/data/shot_usageNEW.csv')
        required_columns = ['shot', 'used_as', 'used_for_ris1', 'used_for_ris2']
        
        if all(col in shot_usage.columns for col in required_columns):
            logger.info(f"✓ Shot usage file has correct structure")
            logger.info(f"  Total shots: {len(shot_usage)}")
            logger.info(f"  RIS1 available: {shot_usage['used_for_ris1'].sum()}")
            logger.info(f"  RIS2 available: {shot_usage['used_for_ris2'].sum()}")
            logger.info(f"  Both available: {(shot_usage['used_for_ris1'] & shot_usage['used_for_ris2']).sum()}")
        else:
            logger.error(f"✗ Shot usage file missing columns: {required_columns}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Failed to read shot usage file: {e}")
        return False
    
    # Test 3: Check if sample LH_alpha files exist
    logger.info("Test 3: Checking LH_alpha data files...")
    try:
        lh_alpha_dir = path / 'data/LH_alpha'
        if not lh_alpha_dir.exists():
            logger.error(f"✗ LH_alpha directory not found: {lh_alpha_dir}")
            return False
            
        # Check for at least one file
        lh_alpha_files = list(lh_alpha_dir.glob('LH_alpha_shot_*.csv'))
        if len(lh_alpha_files) > 0:
            logger.info(f"✓ Found {len(lh_alpha_files)} LH_alpha files")
            
            # Test reading one file
            sample_file = lh_alpha_files[0]
            df_sample = pd.read_csv(sample_file)
            logger.info(f"  Sample file {sample_file.name} has {len(df_sample)} rows")
            logger.info(f"  Columns: {df_sample.columns.tolist()}")
        else:
            logger.error(f"✗ No LH_alpha files found in {lh_alpha_dir}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Failed to check LH_alpha files: {e}")
        return False
    
    logger.info("✓ All tests passed! The both cameras fix should work correctly.")
    return True

if __name__ == '__main__':
    success = test_both_cameras_fix()
    sys.exit(0 if success else 1)
