#!/usr/bin/env python3
"""
Data processing script for COMPASS Tokamak dataset preparation.

This script provides entry points for preparing image and 1D signal data.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_processing.imgs_processing import prepare_dataset_images
from src.data_processing.process_data_for_alt_models import process_data_for_alt_models


def main():
    parser = argparse.ArgumentParser(description='Process COMPASS Tokamak data')
    parser.add_argument('--data-type', choices=['images', 'signals'], 
                       required=True, help='Type of data to process')
    parser.add_argument('--shot-numbers', nargs='+', type=int,
                       help='Shot numbers to process (for signals)')
    parser.add_argument('--variant', default='seidl_2023',
                       help='Variant for signal processing')
    parser.add_argument('--sampling-freq', type=int, default=300,
                       help='Sampling frequency for signals')
    
    args = parser.parse_args()
    
    if args.data_type == 'images':
        print("Processing image data...")
        # Note: imgs_processing functions need to be called individually
        # as they don't have a unified main function
        print("Please use the individual functions from src.data_processing.imgs_processing")
        print("Available functions: load_RIS_data, prepare_dataset_images, etc.")
        
    elif args.data_type == 'signals':
        if not args.shot_numbers:
            print("Error: Shot numbers required for signal processing")
            return
            
        print(f"Processing signal data for shots: {args.shot_numbers}")
        process_data_for_alt_models(
            shot_numbers=args.shot_numbers,
            variant=args.variant,
            sampling_freq=args.sampling_freq
        )
        print("Signal processing completed")


if __name__ == "__main__":
    main()
