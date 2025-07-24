#!/usr/bin/env python3
"""
Main training script for COMPASS Tokamak confinement mode classification.

This script provides a unified entry point for training different model architectures.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.LHmode_classifier import train_and_test_ris_model
from src.training.alt_models_training import train_and_test_alt_model
from src.training.PhyDNet_COMPASS import main as phydnet_main


def main():
    parser = argparse.ArgumentParser(description='Train COMPASS Tokamak classification models')
    parser.add_argument('--model', choices=['resnet', 'alt_models', 'phydnet'], 
                       default='resnet', help='Model type to train')
    parser.add_argument('--ris', choices=['RIS1', 'RIS2', 'both'], 
                       default='RIS1', help='RIS camera option for ResNet models')
    parser.add_argument('--epochs-fc', type=int, default=10, 
                       help='Number of epochs for fine-tuning FC layer')
    parser.add_argument('--epochs-all', type=int, default=20, 
                       help='Number of epochs for fine-tuning all layers')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    if args.model == 'resnet':
        print(f"Training ResNet model on {args.ris} camera data...")
        model, model_path = train_and_test_ris_model(
            ris_option=args.ris,
            num_epochs_for_fc=args.epochs_fc,
            num_epochs_for_all_layers=args.epochs_all,
            batch_size=args.batch_size
        )
        print(f"Model saved to: {model_path}")
        
    elif args.model == 'alt_models':
        print("Training alternative 1D signal models...")
        train_and_test_alt_model()
        
    elif args.model == 'phydnet':
        print("Training PhyDNet model...")
        # PhyDNet uses its own argument parsing, so we just call main
        phydnet_main()


if __name__ == "__main__":
    main()
