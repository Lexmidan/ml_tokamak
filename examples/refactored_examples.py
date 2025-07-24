"""
Example usage of the refactored LHmode_classifier functions.

This script demonstrates various ways to use the modular functions
for different training scenarios.
"""

from pathlib import Path
import torch
import torchvision
from torchvision.models.resnet import ResNet34_Weights, ResNet50_Weights

from src.training.LHmode_classifier import (
    train_and_test_ris_model,
    create_model_from_config,
    load_shot_data,
    create_dataloaders,
    setup_model,
    train_phase,
    test_and_save_results
)


def example_basic_training():
    """Example: Basic model training with default parameters."""
    print("Running basic training example...")
    
    model, model_path = train_and_test_ris_model(
        ris_option='RIS1',
        num_epochs_for_fc=3,
        num_epochs_for_all_layers=3,
        test_run=True  # Use small dataset for quick testing
    )
    
    print(f"Model saved to: {model_path}")


def example_custom_model():
    """Example: Training with a different model architecture."""
    print("Running custom model training example...")
    
    # Create a ResNet34 model
    custom_model = create_model_from_config('resnet34')
    
    model, model_path = train_and_test_ris_model(
        ris_option='RIS1',
        pretrained_model=custom_model,
        num_epochs_for_fc=3,
        num_epochs_for_all_layers=3,
        batch_size=16,
        grayscale=True,
        test_run=True
    )
    
    print(f"Custom model saved to: {model_path}")


def example_manual_training_pipeline():
    """Example: Manual control over the training pipeline."""
    print("Running manual training pipeline example...")
    
    # Setup
    path = Path('.')
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load data
    shots_testing, shots_validation, shots_training = load_shot_data(
        path, 'RIS1', test_df_contains_val_df=True, 
        test_run=True, data_frac=1.0, random_seed=42)
    
    # Create dataloaders
    dataloaders, dataset_sizes, test_dataloader = create_dataloaders(
        path, shots_training, shots_testing, shots_validation,
        'RIS1', num_classes=3, exponential_elm_decay=True,
        batch_size=16, num_workers=4, augmentation=False, grayscale=False)
    
    # Setup model
    pretrained_model = torchvision.models.resnet18()
    model = setup_model(pretrained_model, num_classes=3, device=device, grayscale=False)
    
    # Train only FC layer
    model = train_phase(
        model, dataloaders, dataset_sizes, 
        timestamp='manual_example', path=path, phase='last_fc',
        num_epochs=2, learning_rate_min=0.001, learning_rate_max=0.01,
        weight_decay=1e-4, freeze_backbone=True)
    
    print("Manual training pipeline completed!")


def example_hyperparameter_sweep():
    """Example: Simple hyperparameter sweep."""
    print("Running hyperparameter sweep example...")
    
    learning_rates = [0.001, 0.01]
    batch_sizes = [16, 32]
    
    results = []
    
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"Training with LR={lr}, BS={bs}")
            
            try:
                model, model_path = train_and_test_ris_model(
                    ris_option='RIS1',
                    learning_rate_max=lr,
                    batch_size=bs,
                    num_epochs_for_fc=2,
                    num_epochs_for_all_layers=2,
                    test_run=True,
                    comment_for_model_name=f'_lr{lr}_bs{bs}'
                )
                
                results.append({
                    'lr': lr,
                    'batch_size': bs,
                    'model_path': model_path,
                    'success': True
                })
                
            except Exception as e:
                print(f"Failed with LR={lr}, BS={bs}: {e}")
                results.append({
                    'lr': lr,
                    'batch_size': bs,
                    'error': str(e),
                    'success': False
                })
    
    print("Hyperparameter sweep completed!")
    for result in results:
        if result['success']:
            print(f"✓ LR={result['lr']}, BS={result['batch_size']}: {result['model_path']}")
        else:
            print(f"✗ LR={result['lr']}, BS={result['batch_size']}: {result['error']}")


if __name__ == '__main__':
    print("LHmode_classifier Refactored Examples")
    print("=" * 50)
    
    # Run examples (uncomment the ones you want to try)
    
    # Basic training
    example_basic_training()
    
    # Custom model
    # example_custom_model()
    
    # Manual pipeline control
    # example_manual_training_pipeline()
    
    # Hyperparameter sweep
    # example_hyperparameter_sweep()
    
    print("\nAll examples completed!")
