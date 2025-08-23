#!/usr/bin/env python3
"""
Main training script for COMPASS Tokamak confinement mode classification.

This script provides a unified entry point for training different model architectures.
"""

import sys
import argparse
from pathlib import Path
import random

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.LHmode_classifier import train_and_test_ris_model
from src.training.alt_models_training import train_and_test_alt_model
from src.training.PhyDNet_COMPASS import train_and_eval_PhyDNet
from src.training.PhyDNet_finetuning import finetune_phydnet
from src.utils.utils import gen_run_name

def main():
    print('#####WARNING: argument is implemented for resnet models only#####') #TODO: implement for other models
    default_comment = gen_run_name()
    parser = argparse.ArgumentParser(description='Train COMPASS Tokamak classification models')
    parser.add_argument('--model', choices=['resnet', 'alt_models', 'phydnet', 'phydnet_finetune'], 
                       default='resnet', help='Model type to train')
    parser.add_argument('--ris', choices=['RIS1', 'RIS2', 'both'], 
                       default='RIS1', help='RIS camera option for ResNet models')
    parser.add_argument('--epochs_resnet_fc', type=int, default=10, 
                       help='Number of epochs for fine-tuning FC layer')
    parser.add_argument('--epochs_resnet_all', type=int, default=20, 
                       help='Number of epochs for fine-tuning all layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for data loading')

    parser.add_argument('--comment', type=str, default=default_comment, 
                       help='Comment for the training run')
    

    args = parser.parse_args()
    
    if args.model == 'resnet':
        print(f"Training ResNet model on {args.ris} camera data...")
        model, model_path = train_and_test_ris_model(
            ris_option=args.ris,
            num_epochs_for_fc=args.epochs_resnet_fc,
            num_epochs_for_all_layers=args.epochs_all,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            comment_for_model_name=args.comment
        )
        print(f"Model saved to: {model_path}")
        
    elif args.model == 'alt_models':
        print("Training alternative 1D signal models...")
        train_and_test_alt_model(signal_name = 'mc_h_alpha',
                                 architecture = 'Simple1DCNN',
                                 signal_window = 320,
                                 dpoints_in_future = 160,
                                 sampling_freq = 300,
                                 batch_size = args.batch_size,
                                 num_workers = args.num_workers,
                                 num_epochs = args.epochs_all,
                                 learning_rate_min = 0.001,
                                 learning_rate_max = 0.01,
                                 comment_for_model_name = args.comment,
                                 random_seed = 42,
                                 exponential_elm_decay = False,
                                 num_classes = 3,
                                 weight_decay = 1e-4,
                                 use_ELMs = True,
                                 no_L_mode = False,
                                 only_ELMs = False,
                                 test_df_contains_val_df=True,
                                 test_run=False)
        
    elif args.model == 'phydnet':
        print(f"Training PhyDNet model for --epochs_all {args.epochs_all}")

        train_and_eval_PhyDNet(epochs_all=args.epochs_all, 
                               ris_option=args.ris,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers
                               )
        
    elif args.model == 'phydnet_finetune':
        print(f"Fine-tuning PhyDNet model for --epochs_all {args.epochs_all}")

        finetune_phydnet(epochs_all=args.epochs_all, 
                         ris_option=args.ris,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers
                         )


if __name__ == "__main__":
    main()
