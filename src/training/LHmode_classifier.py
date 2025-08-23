import os
import re
import time 
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, Union

import matplotlib.pyplot as plt
from torch import cuda
import torchvision
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pytorch_lightning as pl
from torchvision.models.resnet import ResNet50_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights, ResNet18_Weights

from ..utils import confinement_mode_classifier as cmc
from ..utils.utils import get_project_root, gen_run_name


def setup_logging(log_dir: Path, phase: str = None) -> logging.Logger:
    """
    Setup logging to save logs to the model directory.
    
    Args:
        log_dir: Directory where logs should be saved
        phase: Training phase (e.g., 'last_fc', 'all_layers')
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('LHmode_classifier')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    log_file = log_dir / f'training_log_{phase}.log' if phase else log_dir / 'training_log.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_shot_data(ris_option: str, test_df_contains_val_df: bool = True, 
                  test_run: bool = False, data_frac: float = 1.0, 
                  random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split shot data for training, validation, and testing.
    
    Args:
        ris_option: 'RIS1', 'RIS2', or 'both'
        test_df_contains_val_df: Whether to include validation shots in test set
        test_run: Whether to run with limited data for testing
        data_frac: Fraction of training data to use
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (shots_for_testing, shots_for_validation, shots_for_training, combined_data)
    """
    logger = logging.getLogger('LHmode_classifier')
    logger.info(f"Loading shot data for {ris_option} option...")
    
    # Automatically detect project root
    path = get_project_root()
    
    shot_usage = pd.read_csv(f'{path}/data/shot_usageNEW.csv')
    
    # Load RIS1 data
    shot_for_ris = shot_usage[shot_usage['used_for_ris1']]
    if ris_option == 'RIS2':
        shot_for_ris = shot_usage[shot_usage['used_for_ris2']]
    
    shots_for_testing = shot_for_ris[shot_for_ris['used_as'] == 'test']['shot']
    shots_for_validation = shot_for_ris[shot_for_ris['used_as'] == 'val']['shot']
    shots_for_training = shot_for_ris[shot_for_ris['used_as'] == 'train']['shot']
    
    if test_df_contains_val_df:
        shots_for_testing = pd.concat([shots_for_testing, shots_for_validation])
    
    if test_run:
        shots_for_testing = shots_for_testing[:3]
        shots_for_validation = shots_for_validation[:3]
        shots_for_training = shots_for_training[:3]
        logger.info("Running in test mode with limited data (3 shots each)")
    
    shots_for_training = shots_for_training.sample(frac=data_frac, random_state=random_seed)
    
    logger.info(f"Data split complete - Training: {len(shots_for_training)}, "
                f"Validation: {len(shots_for_validation)}, Testing: {len(shots_for_testing)}")
    
    return shots_for_testing, shots_for_validation, shots_for_training


def create_dataloaders(shots_for_training: pd.DataFrame, shots_for_testing: pd.DataFrame,
                      shots_for_validation: pd.DataFrame, ris_option: str, num_classes: int,
                      exponential_elm_decay: bool, batch_size: int, num_workers: int,
                      augmentation: bool, grayscale: bool) -> Tuple[Dict, Dict]:
    """
    Create training, validation, and test dataloaders.
    
    Returns:
        Tuple of (dataloaders dict, dataset_sizes dict, test_dataloader)
    """
    logger = logging.getLogger('LHmode_classifier')
    logger.info(f"Creating dataloaders with batch_size={batch_size}, num_workers={num_workers}")
    logger.info(f"Configuration: ris_option={ris_option}, num_classes={num_classes}, "
                f"augmentation={augmentation}, grayscale={grayscale}")
    
    # Automatically detect project root
    path = get_project_root()
    
    shot_numbers = pd.concat([shots_for_training, shots_for_testing, shots_for_validation])
    
    # Handle the 'both' case properly
    if ris_option == 'both':
        logger.info("Loading data for both RIS1 and RIS2...")
        # For 'both', we need to load RIS1 data first, then combine with RIS2
        shot_df, test_df, val_df, train_df = cmc.load_and_split_dataframes(
            path, shot_numbers, shots_for_training, shots_for_testing, 
            shots_for_validation, use_ELMS=num_classes==3, ris_option='RIS1',
            exponential_elm_decay=exponential_elm_decay)
        
        # Load RIS2 data and combine
        shot_df_ris2, test_df_ris2, val_df_ris2, train_df_ris2 = cmc.load_and_split_dataframes(
            path, shot_numbers, shots_for_training, shots_for_testing, 
            shots_for_validation, use_ELMS=num_classes==3, ris_option='RIS2',
            exponential_elm_decay=exponential_elm_decay)
        
        test_df = pd.concat([test_df, test_df_ris2]).reset_index(drop=True)
        val_df = pd.concat([val_df, val_df_ris2]).reset_index(drop=True)
        train_df = pd.concat([train_df, train_df_ris2]).reset_index(drop=True)
        logger.info("Combined RIS1 and RIS2 data successfully")
    else:
        logger.info(f"Loading data for {ris_option}...")
        shot_df, test_df, val_df, train_df = cmc.load_and_split_dataframes(
            path, shot_numbers, shots_for_training, shots_for_testing, 
            shots_for_validation, use_ELMS=num_classes==3, ris_option=ris_option,
            exponential_elm_decay=exponential_elm_decay)
    
    logger.info("Creating dataloaders...")
    test_dataloader = cmc.get_dloader(test_df, path, batch_size, balance_data=False, 
                                      shuffle=False, num_workers=num_workers, 
                                      augmentation=False, grayscale=grayscale)

    val_dataloader = cmc.get_dloader(val_df, path, batch_size, balance_data=True, 
                                     shuffle=False, num_workers=num_workers, 
                                     augmentation=False, grayscale=grayscale)

    train_dataloader = cmc.get_dloader(train_df, path, batch_size, balance_data=True, 
                                       shuffle=False, num_workers=num_workers, 
                                       augmentation=augmentation, grayscale=grayscale)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    
    logger.info(f"Dataloaders created successfully - Train: {dataset_sizes['train']} samples, "
                f"Val: {dataset_sizes['val']} samples, Test: {len(test_dataloader.dataset)} samples")
    
    return dataloaders, dataset_sizes, test_dataloader


def prepare_model_for_grayscale(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Modify the first convolutional layer of a pretrained model to accept grayscale input.
    """
    # Luminance weights for RGB to grayscale conversion
    weights_rgb_to_gray = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(device)

    # Get the original weights of the first conv layer
    original_weights = model.conv1.weight.data

    # Compute the weighted sum of the RGB channels
    grayscale_weights = (original_weights * weights_rgb_to_gray).sum(dim=1, keepdim=True)

    # Update the first convolutional layer
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=model.conv1.bias is not None)

    # Assign the new grayscale weights to the first conv layer
    model.conv1.weight = nn.Parameter(grayscale_weights)

    # If there is a bias term, keep it unchanged
    if model.conv1.bias is not None:
        model.conv1.bias = nn.Parameter(model.conv1.bias.data)
    
    return model


def setup_model(pretrained_model: nn.Module, num_classes: int, device: torch.device, 
                grayscale: bool = False) -> nn.Module:
    """
    Setup the pretrained model for transfer learning.
    """
    logger = logging.getLogger('LHmode_classifier')
    logger.info(f"Setting up model for {num_classes} classes on device: {device}")
    
    # Freeze all parameters initially
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, num_classes)
    pretrained_model = pretrained_model.to(device)

    if grayscale:
        logger.info("Adapting model for grayscale input")
        pretrained_model = prepare_model_for_grayscale(pretrained_model, device)
    
    logger.info("Model setup complete")
    return pretrained_model


def create_optimizer_and_scheduler(model: nn.Module, learning_rate_min: float, 
                                  learning_rate_max: float, weight_decay: float,
                                  dataset_size: int, num_epochs: int):
    """
    Create optimizer and learning rate scheduler.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_min, weight_decay=weight_decay)
    exp_lr_scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate_max, 
        steps_per_epoch=dataset_size, epochs=num_epochs)
    
    return optimizer, exp_lr_scheduler


def save_hyperparameters_and_metrics(path: Path, timestamp: str, phase: str, 
                                    hyperparameters: Dict, metrics: Dict, writer: SummaryWriter,
                                    base_dir: str = 'runs'):
    """
    Save hyperparameters and metrics to JSON file and TensorBoard.
    """
    # Create a copy of hyperparameters for TensorBoard (excluding problematic keys)
    if not timestamp:
        timestamp = gen_run_name()
    tb_hyperparameters = {}
    json_hyperparameters = {}
    
    for key, value in hyperparameters.items():
        # Convert tensors and arrays to lists for JSON serialization
        if hasattr(value, 'tolist'):
            json_hyperparameters[key] = value.tolist()
        else:
            json_hyperparameters[key] = value
        
        # Only include simple types for TensorBoard
        if key not in ['shots_for_testing', 'shots_for_validation', 'shots_for_training']:
            if isinstance(value, (int, float, str, bool)) or (hasattr(value, 'item') and callable(getattr(value, 'item'))):
                if hasattr(value, 'item'):
                    tb_hyperparameters[key] = value.item()
                else:
                    tb_hyperparameters[key] = value
    
    one_digit_metrics = {
        'Accuracy on test_dataset': metrics['accuracy'], 
        'F1 metric on test_dataset': metrics['f1'], 
        'Precision on test_dataset': metrics['precision'], 
        'Recall on test_dataset': metrics['recall']
    }
    
    writer.add_hparams(tb_hyperparameters, one_digit_metrics)
    
    # Save to JSON
    all_hparams = {**json_hyperparameters, **one_digit_metrics}
    json_str = json.dumps(all_hparams, indent=4)
    
    # Ensure the directory exists
    json_path = path / base_dir / f'{timestamp}_{phase}' / 'hparams.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w') as f:
        f.write(json_str)


def train_phase(model: nn.Module, dataloaders: Dict, dataset_sizes: Dict, 
               timestamp: str, phase: str, num_epochs: int,
               learning_rate_min: float, learning_rate_max: float, 
               weight_decay: float, freeze_backbone: bool = True,
               base_dir: str = 'runs') -> nn.Module:
    """
    Train the model for a specific phase (either just FC layer or all layers).
    
    Args:
        model: The model to train
        dataloaders: Dictionary containing train and val dataloaders
        dataset_sizes: Dictionary containing dataset sizes
        timestamp: Timestamp for model naming
        phase: 'last_fc' or 'all_layers'
        num_epochs: Number of epochs to train
        learning_rate_min: Minimum learning rate
        learning_rate_max: Maximum learning rate
        weight_decay: Weight decay for optimizer
        freeze_backbone: Whether to freeze backbone (True for FC-only training)
        base_dir: Base directory for saving models and logs (default: 'runs')
        
    Returns:
        Trained model
    """
    logger = logging.getLogger('LHmode_classifier')
    phase_name = "FC layer only" if phase == 'last_fc' else "all layers"
    logger.info(f"Starting training phase: {phase_name} for {num_epochs} epochs")
    logger.info(f"Learning rate range: {learning_rate_min} to {learning_rate_max}, weight_decay: {weight_decay}")
    
    # Automatically detect project root and ensure the run directory exists
    path = get_project_root()
    run_dir = path / base_dir / f'{timestamp}_{phase}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(str(run_dir))
    
    # Set parameter gradients based on training phase
    for param in model.parameters():
        param.requires_grad = not freeze_backbone
    
    # Always allow gradients for the final layer
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer, exp_lr_scheduler = create_optimizer_and_scheduler(
        model, learning_rate_min, learning_rate_max, weight_decay,
        dataset_sizes['train'], num_epochs)
    
    # Model save paths
    model_path = run_dir / 'model.pt'
    chkpt_path = run_dir / f'model_best_val_acc.pt'
    
    logger.info(f"Model will be saved to: {model_path}")
    
    # Train the model
    logger.info("Starting model training...")
    model = cmc.train_model(
        model, criterion, optimizer, exp_lr_scheduler, 
        dataloaders, writer, dataset_sizes, num_epochs=num_epochs, 
        chkpt_path=chkpt_path,
        return_best_model=False)
    
    # Save model
    torch.save(model.state_dict(), model_path)
    logger.info(f"Training phase {phase_name} completed and model saved")
    
    writer.close()
    return model


def test_and_save_results(model: nn.Module, test_dataloader, timestamp: str, 
                         phase: str, shots_for_testing: pd.DataFrame, num_classes: int,
                         ris_option: str, hyperparameters: Dict, base_dir: str = 'runs'):
    """
    Test the model and save results including predictions and metrics.
    """
    logger = logging.getLogger('LHmode_classifier')
    logger.info(f"Starting model testing for phase: {phase}")
    if not timestamp:
        timestamp = gen_run_name()
    # Automatically detect project root and ensure the run directory exists
    path = get_project_root()
    run_dir = path / base_dir / f'{timestamp}_{phase}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(str(run_dir))
    
    # Test the model
    logger.info("Running model evaluation on test dataset...")
    metrics = cmc.test_model(
        str(run_dir), model, test_dataloader, comment='', 
        writer=writer, num_classes=num_classes, signal_name='img')

    # Save predictions
    prediction_path = run_dir / 'prediction_df.csv'
    metrics['prediction_df'].to_csv(prediction_path)
    logger.info(f"Predictions saved to: {prediction_path}")

    # Per-shot analysis
    logger.info("Performing per-shot analysis...")
    metrics_per_shot = cmc.per_shot_test(
        path=str(run_dir) + '/', 
        shots=shots_for_testing.values.tolist(), 
        results_df=metrics['prediction_df'], 
        writer=writer,
        num_classes=num_classes,
        two_images=ris_option=='both')

    metrics_path = run_dir / 'metrics_per_shot.csv'
    pd.DataFrame(metrics_per_shot).to_csv(metrics_path)
    logger.info(f"Per-shot metrics saved to: {metrics_path}")
    
    # Save hyperparameters and metrics
    logger.info("Saving hyperparameters and final metrics...")
    save_hyperparameters_and_metrics(path, timestamp, phase, hyperparameters, metrics, writer, base_dir)
    
    # Log key metrics
    logger.info(f"Test Results - Accuracy: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1']:.4f}")
    
    writer.close()
    return metrics


def create_model_from_config(model_name: str, weights: Optional[str] = None) -> nn.Module:
    """
    Create a model from configuration string.
    
    Args:
        model_name: Name of the model ('resnet18', 'resnet34', etc.)
        weights: Weights to use (if None, uses default pretrained weights)
        
    Returns:
        Pretrained model
    """
    model_mapping = {
        'resnet18': (torchvision.models.resnet18, ResNet18_Weights.IMAGENET1K_V1),
        'resnet34': (torchvision.models.resnet34, ResNet34_Weights.IMAGENET1K_V1),
        'resnet50': (torchvision.models.resnet50, ResNet50_Weights.IMAGENET1K_V1),
        'resnet101': (torchvision.models.resnet101, ResNet101_Weights.IMAGENET1K_V1),
        'resnet152': (torchvision.models.resnet152, ResNet152_Weights.IMAGENET1K_V1),
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Model {model_name} not supported. Available: {list(model_mapping.keys())}")
    
    model_fn, default_weights = model_mapping[model_name]
    weights_to_use = weights or default_weights
    
    return model_fn(weights=weights_to_use)

def train_and_test_ris_model(ris_option: str = 'both',
                            pretrained_model=torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
                            num_workers: int = 32,
                            num_epochs_for_fc: int = 10,
                            num_epochs_for_all_layers: int = 10,
                            num_classes: int = 3,
                            batch_size: int = 32,
                            learning_rate_min: float = 0.001,
                            learning_rate_max: float = 0.01,
                            comment_for_model_name: str = ', 3 output classes',
                            random_seed: int = 42,
                            augmentation: bool = False,
                            test_df_contains_val_df: bool = True,
                            test_run: bool = False,
                            exponential_elm_decay: bool = True,
                            grayscale: bool = False,
                            weight_decay: float = 1e-4,
                            data_frac: float = 1.0) -> Tuple[nn.Module, Path]:
    """
    Trains a RIS model with transfer learning in two phases:
    1. Train only the final fully connected layer
    2. Fine-tune all layers
    
    Args:
        ris_option: 'RIS1', 'RIS2', or 'both'
        pretrained_model: Pretrained model to use as backbone
        num_workers: Number of workers for data loading
        num_epochs_for_fc: Epochs for training only FC layer
        num_epochs_for_all_layers: Epochs for fine-tuning all layers
        num_classes: Number of output classes
        batch_size: Batch size for training
        learning_rate_min: Minimum learning rate
        learning_rate_max: Maximum learning rate for OneCycle scheduler
        comment_for_model_name: Comment to add to model name
        random_seed: Random seed for reproducibility
        augmentation: Whether to apply data augmentation
        test_df_contains_val_df: Whether to include validation shots in test set
        test_run: Whether to run with limited data for testing
        exponential_elm_decay: Whether to apply exponential ELM decay
        grayscale: Whether to use grayscale images
        weight_decay: Weight decay for optimizer
        data_frac: Fraction of training data to use
        
    Returns:
        Tuple of (trained_model, model_path)
    """
    # Setup
    comment_for_model_name = ris_option + comment_for_model_name
    pl.seed_everything(random_seed)
    path = get_project_root()  # Use project root instead of current working directory
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    timestamp = datetime.fromtimestamp(time.time()).strftime("%y-%m-%d, %H-%M-%S ") + comment_for_model_name

    # Setup logging - create a temporary logger for initial setup
    temp_log_dir = path / 'runs' / f'{timestamp}_setup'
    logger = setup_logging(temp_log_dir)
    
    logger.info("="*60)
    logger.info("STARTING L-H MODE CLASSIFIER TRAINING")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  - RIS option: {ris_option}")
    logger.info(f"  - Model: {pretrained_model.__class__.__name__}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Timestamp: {timestamp}")
    logger.info(f"  - Epochs (FC/All): {num_epochs_for_fc}/{num_epochs_for_all_layers}")
    logger.info(f"  - Batch size: {batch_size}, Workers: {num_workers}")
    logger.info(f"  - Learning rate: {learning_rate_min} to {learning_rate_max}")

    # Load and prepare data
    logger.info("Phase 1/5: Loading and preparing data...")
    shots_for_testing, shots_for_validation, shots_for_training = load_shot_data(
        'RIS1' if ris_option == 'both' else ris_option, 
        test_df_contains_val_df, test_run, data_frac, random_seed)
    
    # Handle 'both' option by combining RIS1 and RIS2 data
    if ris_option == 'both':
        logger.info("Loading additional RIS2 data for 'both' option...")
        shots_for_testing_ris2, shots_for_validation_ris2, shots_for_training_ris2 = load_shot_data(
            'RIS2', test_df_contains_val_df, test_run, data_frac, random_seed)
        
        shots_for_testing = pd.concat([shots_for_testing, shots_for_testing_ris2]).reset_index(drop=True)
        shots_for_validation = pd.concat([shots_for_validation, shots_for_validation_ris2]).reset_index(drop=True)
        shots_for_training = pd.concat([shots_for_training, shots_for_training_ris2]).reset_index(drop=True)
        logger.info("Combined RIS1 and RIS2 shot data")

    # Create dataloaders
    logger.info("Phase 2/5: Creating dataloaders...")
    dataloaders, dataset_sizes, test_dataloader = create_dataloaders(
        shots_for_training, shots_for_testing, shots_for_validation,
        ris_option, num_classes, exponential_elm_decay, batch_size, 
        num_workers, augmentation, grayscale)

    # Setup model
    logger.info("Phase 3/5: Setting up model...")
    model = setup_model(pretrained_model, num_classes, device, grayscale)

    # Create base hyperparameters dictionary
    base_hyperparameters = {
        'model': model.__class__.__name__,
        'batch_size': batch_size,
        'ris_option': ris_option,
        'num_classes': num_classes,
        'second_image': 'None',
        'augmentation': "applied" if augmentation else "no augmentation",
        'random_seed': random_seed,
        'weight_decay': weight_decay,
        'shots_for_testing': shots_for_testing.values.tolist(),
        'shots_for_validation': shots_for_validation.values.tolist(),
        'shots_for_training': shots_for_training.values.tolist(),
    }

    # Phase 1: Train only the fully connected layer
    logger.info("Phase 4a/5: Training fully connected layer...")
    logger.info("-" * 40)
    
    # Setup logging for FC phase
    fc_log_dir = Path(f'{path}/runs/{timestamp}_last_fc')
    setup_logging(fc_log_dir, 'last_fc')
    
    model = train_phase(
        model, dataloaders, dataset_sizes, timestamp, path, 'last_fc',
        num_epochs_for_fc, learning_rate_min, learning_rate_max, weight_decay,
        freeze_backbone=True)

    # Test after FC training
    fc_hyperparameters = {
        **base_hyperparameters,
        'num_epochs': num_epochs_for_fc,
        'optimizer': 'AdamW',
        'criterion': 'CrossEntropyLoss',
        'learning_rate_max': learning_rate_max,
        'scheduler': 'OneCycleLR',
    }
    
    logger.info("Testing FC-only model...")
    test_and_save_results(
        model, test_dataloader, path, timestamp, 'last_fc',
        shots_for_testing, num_classes, ris_option, fc_hyperparameters)

    # Phase 2: Fine-tune all layers
    logger.info("Phase 4b/5: Fine-tuning all layers...")
    logger.info("-" * 40)
    torch.cuda.empty_cache()
    
    # Setup logging for all layers phase
    all_layers_log_dir = Path(f'{path}/runs/{timestamp}_all_layers')
    setup_logging(all_layers_log_dir, 'all_layers')
    
    model = train_phase(
        model, dataloaders, dataset_sizes, timestamp, path, 'all_layers',
        num_epochs_for_all_layers, learning_rate_min, learning_rate_max, weight_decay,
        freeze_backbone=False)

    # Test after full training
    all_layers_hyperparameters = {
        **base_hyperparameters,
        'num_epochs': num_epochs_for_all_layers,
        'optimizer': 'AdamW',
        'criterion': 'CrossEntropyLoss',
        'learning_rate_max': learning_rate_max,
        'scheduler': 'OneCycleLR',
    }
    
    logger.info("Testing full model...")
    test_and_save_results(
        model, test_dataloader, path, timestamp, 'all_layers',
        shots_for_testing, num_classes, ris_option, all_layers_hyperparameters)

    model_path = Path(f'{path}/runs/{timestamp}_all_layers/model.pt')
    
    logger.info("Phase 5/5: Training completed!")
    logger.info("="*60)
    logger.info(f"TRAINING COMPLETE - Model saved to: {model_path}")
    logger.info("="*60)
    
    return model, model_path

if __name__ == '__main__':
    # Example usage with default parameters
    #model, model_path = train_and_test_ris_model()
    
    # Example of using different configurations:
    model, model_path = train_and_test_ris_model(
        ris_option='RIS1',
        pretrained_model=create_model_from_config('resnet18'),
        num_workers=4,
        num_epochs_for_fc=1,
        num_epochs_for_all_layers=1,
        batch_size=64,
        learning_rate_min=1e-4,
        learning_rate_max=1e-3,
        weight_decay=1e-4,
        comment_for_model_name=' resnet18, 3 output classes',
        random_seed=42,
        augmentation=False,
        test_df_contains_val_df=True,
        test_run=False,
        exponential_elm_decay=False,
        grayscale=False,
        data_frac=1
    )
    print(f'Training completed. Model saved to: {model_path}')
