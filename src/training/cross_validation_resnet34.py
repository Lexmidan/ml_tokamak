import os
import re
import time 
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, Union, List
import traceback

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
from sklearn.model_selection import KFold
import numpy as np

from . import LHmode_classifier as LH
from ..utils import confinement_mode_classifier as cmc


def setup_cv_logging(log_dir: Path, fold_idx: int = None) -> logging.Logger:
    """
    Setup logging for cross-validation that saves to results directory.
    
    Args:
        log_dir: Directory where logs should be saved
        fold_idx: Fold index (if None, creates general CV log)
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger_name = 'cross_validation'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    if fold_idx is not None:
        log_file = log_dir / f'fold_{fold_idx + 1}_log.log'
    else:
        log_file = log_dir / 'cross_validation_log.log'
    
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


def create_dataloaders_for_both_cameras(path: Path, shots_for_training: pd.DataFrame, shots_for_testing: pd.DataFrame,
                                       shots_for_validation: pd.DataFrame, num_classes: int,
                                       exponential_elm_decay: bool, batch_size: int, num_workers: int,
                                       augmentation: bool, grayscale: bool, test_run: bool = False) -> Tuple[Dict, Dict]:
    """
    Create dataloaders for 'both' cameras, properly handling camera availability per shot.
    
    Returns:
        Tuple of (dataloaders dict, dataset_sizes dict, test_dataloader)
    """
    logger = logging.getLogger('cross_validation')
    logger.info("Creating dataloaders for 'both' cameras with proper camera availability handling...")
    
    # Load shot usage to determine camera availability
    shot_usage = pd.read_csv(f'{path}/data/shot_usageNEW.csv')
    shot_usage_dict = shot_usage.set_index('shot')[['used_for_ris1', 'used_for_ris2']].to_dict('index')
    
    # Combine all shots
    all_shots = pd.concat([shots_for_training, shots_for_testing, shots_for_validation]).unique()
    
    # Load and combine dataframes from available cameras for each shot
    combined_shot_df = pd.DataFrame()
    
    for shot in all_shots:
        if shot not in shot_usage_dict:
            logger.warning(f"Shot {shot} not found in shot_usageNEW.csv, skipping...")
            continue
            
        shot_info = shot_usage_dict[shot]
        
        # Load RIS1 data if available
        if shot_info['used_for_ris1']:
            try:
                df_ris1 = pd.read_csv(f'{path}/data/LH_alpha/LH_alpha_shot_{shot}.csv')
                df_ris1['shot'] = shot
                df_ris1 = df_ris1.iloc[:-100]  # Drop last 100 rows
                combined_shot_df = pd.concat([combined_shot_df, df_ris1], axis=0)
            except FileNotFoundError:
                logger.warning(f"RIS1 data file not found for shot {shot}")
        
        # Load RIS2 data if available
        if shot_info['used_for_ris2']:
            try:
                df_ris2 = pd.read_csv(f'{path}/data/LH_alpha/LH_alpha_shot_{shot}.csv')
                df_ris2['shot'] = shot
                df_ris2 = df_ris2.iloc[:-100]  # Drop last 100 rows
                # Convert RIS1 filenames to RIS2
                df_ris2['filename'] = df_ris2['filename'].str.replace('RIS1', 'RIS2')
                
                # Filter out rows where RIS2 image doesn't exist
                def image_exists(filename):
                    return os.path.exists(os.path.join(path, filename))
                
                exists_mask = df_ris2['filename'].apply(image_exists)
                df_ris2 = df_ris2[exists_mask]
                
                combined_shot_df = pd.concat([combined_shot_df, df_ris2], axis=0)
            except FileNotFoundError:
                logger.warning(f"RIS2 data file not found for shot {shot}")
    
    # Apply exponential ELM decay if needed
    if exponential_elm_decay and num_classes == 3:
        # Apply the same ELM decay logic as in confinement_mode_classifier
        combined_shot_df['soft_label'] = combined_shot_df.apply(
            lambda x: [0, 1, 0] if x['mode'] == 'H-mode' else [1, 0, 0], axis=1)
        
        # Pre and post peak time processing for ELMs
        pre_time = 1
        post_time = 2
        
        for shot in combined_shot_df['shot'].unique():
            shot_data = combined_shot_df[combined_shot_df['shot'] == shot]
            
            if shot_data['mode'].str.contains('ELM-peak').any():
                for elm_peak in shot_data[shot_data['mode'] == 'ELM-peak']['time']:
                    # Pre-ELM probabilities
                    pre_indices = shot_data.loc[shot_data['time'].between(elm_peak-pre_time, elm_peak)].index
                    if not pre_indices.empty:
                        pre_elm_prob = np.exp(-5 * np.linspace(pre_time, 0, len(pre_indices)))
                        for i, prob in zip(pre_indices, pre_elm_prob):
                            combined_shot_df.at[i, 'soft_label'] = [0, 1 - np.max([prob, combined_shot_df.at[i, 'soft_label'][2]]), 
                                                        np.max([prob, combined_shot_df.at[i, 'soft_label'][2]])]
                    
                    # Post-ELM probabilities
                    post_indices = shot_data.loc[shot_data['time'].between(elm_peak, elm_peak+post_time)].index
                    if not post_indices.empty:
                        post_elm_prob = np.exp(-3 * np.linspace(0, post_time, len(post_indices)))
                        for i, prob in zip(post_indices, post_elm_prob):
                            combined_shot_df.at[i, 'soft_label'] = [0, 1 - np.max([prob, combined_shot_df.at[i, 'soft_label'][2]]), 
                                                        np.max([prob, combined_shot_df.at[i, 'soft_label'][2]])]

    # Convert mode labels to numeric
    df_mode = combined_shot_df['mode'].copy()
    df_mode[combined_shot_df['mode']=='L-mode'] = 0
    df_mode[combined_shot_df['mode']=='H-mode'] = 1
    df_mode[combined_shot_df['mode']=='ELM'] = 2 if num_classes == 3 else 1
    combined_shot_df['mode'] = df_mode
    combined_shot_df = combined_shot_df.reset_index(drop=True)
    
    # Apply fast test mode sampling if enabled
    if test_run:
        logger.info("ULTRA-FAST MODE: Sampling only 100 images per shot for both cameras...")
        def sample_shot_data(df, max_rows_per_shot=100):
            if len(df) == 0:
                return df
            sampled_dfs = []
            for shot in df['shot'].unique():
                shot_data = df[df['shot'] == shot]
                if len(shot_data) > max_rows_per_shot:
                    shot_data = shot_data.sample(n=max_rows_per_shot, random_state=42)
                sampled_dfs.append(shot_data)
            return pd.concat(sampled_dfs, ignore_index=True)
        
        combined_shot_df = sample_shot_data(combined_shot_df, 100)
        logger.info(f"ULTRA-FAST MODE: Reduced combined data to {len(combined_shot_df)} samples")
    
    # Split into train/val/test dataframes
    test_df = combined_shot_df[combined_shot_df['shot'].isin(shots_for_testing)].reset_index(drop=True)
    val_df = combined_shot_df[combined_shot_df['shot'].isin(shots_for_validation)].reset_index(drop=True)
    train_df = combined_shot_df[combined_shot_df['shot'].isin(shots_for_training)].reset_index(drop=True)
    
    logger.info(f"Combined dataframes created - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create dataloaders
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


def load_shot_data_for_both_cameras(path: Path, test_df_contains_val_df: bool = True, 
                                   test_run: bool = False, data_frac: float = 1.0, 
                                   random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load shot data for 'both' RIS option, considering camera availability for each shot.
    
    Args:
        path: Path to the data directory
        test_df_contains_val_df: Whether to include validation shots in test set
        test_run: Whether to run with limited data for testing
        data_frac: Fraction of training data to use
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (shots_for_testing, shots_for_validation, shots_for_training)
    """
    logger = logging.getLogger('cross_validation')
    logger.info("Loading shot data for 'both' cameras option...")
    
    shot_usage = pd.read_csv(f'{path}/data/shot_usageNEW.csv')
    
    # Get shots that have either RIS1 or RIS2 data (or both)
    shots_with_cameras = shot_usage[
        (shot_usage['used_for_ris1'] == True) | (shot_usage['used_for_ris2'] == True)
    ]
    
    shots_for_testing = shots_with_cameras[shots_with_cameras['used_as'] == 'test']['shot']
    shots_for_validation = shots_with_cameras[shots_with_cameras['used_as'] == 'val']['shot']
    shots_for_training = shots_with_cameras[shots_with_cameras['used_as'] == 'train']['shot']
    
    if test_df_contains_val_df:
        shots_for_testing = pd.concat([shots_for_testing, shots_for_validation])
    
    if test_run:
        shots_for_testing = shots_for_testing[:3]
        shots_for_validation = shots_for_validation[:3]
        shots_for_training = shots_for_training[:3]
        logger.info("Running in test mode with limited data (3 shots each)")
    
    shots_for_training = shots_for_training.sample(frac=data_frac, random_state=random_seed)
    
    logger.info(f"Data split complete for 'both' cameras - Training: {len(shots_for_training)}, "
                f"Validation: {len(shots_for_validation)}, Testing: {len(shots_for_testing)}")
    
    return shots_for_testing, shots_for_validation, shots_for_training


def create_k_fold_splits(shots_df: pd.Series, k: int = 5, random_seed: int = 42) -> List[Tuple[pd.Series, pd.Series]]:
    """
    Create K-fold splits for cross validation.
    
    Args:
        shots_df: Series containing all shots used for training/validation
        k: Number of folds
        random_seed: Random seed for reproducibility
        
    Returns:
        List of tuples (train_shots, val_shots) for each fold
    """
    # Convert to numpy array for sklearn compatibility
    shots_array = shots_df.values
    
    # Initialize KFold
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    
    splits = []
    for train_idx, val_idx in kfold.split(shots_array):
        # Create Series with the proper structure (maintaining shot numbers as Series)
        train_shots = pd.Series(shots_array[train_idx], name='shot')
        val_shots = pd.Series(shots_array[val_idx], name='shot')
        splits.append((train_shots, val_shots))
    
    return splits


def run_single_fold(fold_idx: int, train_shots: pd.Series, val_shots: pd.Series, 
                   test_shots: pd.Series, config: Dict, base_timestamp: str) -> Dict:
    """
    Run training and testing for a single fold.
    
    Args:
        fold_idx: Current fold index
        train_shots: Training shots for this fold
        val_shots: Validation shots for this fold
        test_shots: Test shots (same for all folds)
        config: Training configuration
        base_timestamp: Base timestamp for folder naming
        
    Returns:
        Dictionary containing metrics and results for this fold
    """
    path = Path(os.getcwd())
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # Setup logging for this fold
    cv_results_dir = Path(f'{path}/cross_validation_results/{base_timestamp}')
    logger = setup_cv_logging(cv_results_dir, fold_idx)
    
    logger.info("="*50)
    logger.info(f"Running Fold {fold_idx + 1}/5")
    logger.info("="*50)
    
    # Create fold-specific timestamp that matches the directory structure
    fold_timestamp = f"{base_timestamp}/fold_{fold_idx + 1}"
    
    # Clear GPU memory before creating dataloaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create dataloaders for this fold
    if config['ris_option'] == 'both':
        dataloaders, dataset_sizes, test_dataloader = create_dataloaders_for_both_cameras(
            path, train_shots, test_shots, val_shots,
            config['num_classes'], config['exponential_elm_decay'], 
            config['batch_size'], config['num_workers'], config['augmentation'], config['grayscale'], 
            config.get('test_run', False))
    else:
        dataloaders, dataset_sizes, test_dataloader = LH.create_dataloaders(
            path, train_shots, test_shots, val_shots,
            config['ris_option'], config['num_classes'], config['exponential_elm_decay'], 
            config['batch_size'], config['num_workers'], config['augmentation'], config['grayscale'])

    # Setup model
    pretrained_model = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model = LH.setup_model(pretrained_model, config['num_classes'], device, config['grayscale'])

    # Clear GPU memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 1: Train only the fully connected layer
    logger.info(f"Fold {fold_idx + 1} - Phase 1: Training fully connected layer...")
    model = LH.train_phase(
        model, dataloaders, dataset_sizes, fold_timestamp, path, 'last_fc',
        config['num_epochs_for_fc'], config['learning_rate_min'], 
        config['learning_rate_max'], config['weight_decay'], freeze_backbone=True,
        base_dir='cross_validation_results')

    # Create CV results directory for this fold
    cv_results_dir = Path(f'{path}/cross_validation_results/{base_timestamp}')
    cv_results_dir.mkdir(parents=True, exist_ok=True)
    
    fold_fc_dir = cv_results_dir / f'fold_{fold_idx + 1}_last_fc'
    fold_all_layers_dir = cv_results_dir / f'fold_{fold_idx + 1}_all_layers'
    fold_fc_dir.mkdir(exist_ok=True)
    fold_all_layers_dir.mkdir(exist_ok=True)

    # Test after FC training
    fc_writer = SummaryWriter(f'cross_validation_results/{fold_timestamp}_last_fc')
    fc_metrics = cmc.test_model(
        str(fold_fc_dir), model, test_dataloader, 
        comment=f'fold_{fold_idx + 1}', writer=fc_writer, 
        num_classes=config['num_classes'], signal_name='img')
    
    # Per-shot analysis for FC model
    fc_per_shot_metrics = cmc.per_shot_test(
        path=str(fold_fc_dir) + '/', 
        shots=test_shots.values.tolist(), 
        results_df=fc_metrics['prediction_df'], 
        writer=fc_writer,
        num_classes=config['num_classes'],
        two_images=config['ris_option']=='both')
    
    fc_writer.close()

    # Save predictions and per-shot metrics for FC model
    fc_metrics['prediction_df'].to_csv(fold_fc_dir / 'prediction_df.csv')
    pd.DataFrame(fc_per_shot_metrics).to_csv(fold_fc_dir / 'metrics_per_shot.csv')

    # Phase 2: Fine-tune all layers
    logger.info(f"Fold {fold_idx + 1} - Phase 2: Fine-tuning all layers...")
    torch.cuda.empty_cache()
    
    model = LH.train_phase(
        model, dataloaders, dataset_sizes, fold_timestamp, path, 'all_layers',
        config['num_epochs_for_all_layers'], config['learning_rate_min'], 
        config['learning_rate_max'], config['weight_decay'], freeze_backbone=False,
        base_dir='cross_validation_results')

    # Test after full training
    all_layers_writer = SummaryWriter(f'cross_validation_results/{fold_timestamp}_all_layers')
    all_layers_metrics = cmc.test_model(
        str(fold_all_layers_dir), model, test_dataloader, 
        comment=f'fold_{fold_idx + 1}', writer=all_layers_writer, 
        num_classes=config['num_classes'], signal_name='img')
    
    # Per-shot analysis
    per_shot_metrics = cmc.per_shot_test(
        path=str(fold_all_layers_dir) + '/', 
        shots=test_shots.values.tolist(), 
        results_df=all_layers_metrics['prediction_df'], 
        writer=all_layers_writer,
        num_classes=config['num_classes'],
        two_images=config['ris_option']=='both')
    
    all_layers_writer.close()

    # Save predictions and per-shot metrics
    all_layers_metrics['prediction_df'].to_csv(fold_all_layers_dir / 'prediction_df.csv')
    pd.DataFrame(per_shot_metrics).to_csv(fold_all_layers_dir / 'metrics_per_shot.csv')

    # The model is already saved by train_phase to the correct location
    model_path = Path(f'{path}/cross_validation_results/{fold_timestamp}_all_layers/model.pt')

    # Return results for this fold
    fold_results = {
        'fold_idx': fold_idx,
        'fc_metrics': {
            'accuracy': fc_metrics['accuracy'],
            'f1': fc_metrics['f1'].item() if hasattr(fc_metrics['f1'], 'item') else fc_metrics['f1'],
            'precision': fc_metrics['precision'].item() if hasattr(fc_metrics['precision'], 'item') else fc_metrics['precision'],
            'recall': fc_metrics['recall'].item() if hasattr(fc_metrics['recall'], 'item') else fc_metrics['recall']
        },
        'fc_per_shot_metrics': fc_per_shot_metrics,
        'all_layers_metrics': {
            'accuracy': all_layers_metrics['accuracy'],
            'f1': all_layers_metrics['f1'].item() if hasattr(all_layers_metrics['f1'], 'item') else all_layers_metrics['f1'],
            'precision': all_layers_metrics['precision'].item() if hasattr(all_layers_metrics['precision'], 'item') else all_layers_metrics['precision'],
            'recall': all_layers_metrics['recall'].item() if hasattr(all_layers_metrics['recall'], 'item') else all_layers_metrics['recall']
        },
        'per_shot_metrics': per_shot_metrics,
        'train_shots': train_shots.values.tolist(),
        'val_shots': val_shots.values.tolist(),
        'model_path': str(model_path)
    }
    
    logger.info(f"Fold {fold_idx + 1} completed:")
    logger.info(f"  FC-only - Accuracy: {fold_results['fc_metrics']['accuracy']:.4f}, F1: {fold_results['fc_metrics']['f1']:.4f}")
    logger.info(f"  All layers - Accuracy: {fold_results['all_layers_metrics']['accuracy']:.4f}, F1: {fold_results['all_layers_metrics']['f1']:.4f}")
    
    return fold_results


def aggregate_cv_results(cv_results: List[Dict], config: Dict, base_timestamp: str):
    """
    Aggregate and save cross-validation results.
    
    Args:
        cv_results: List of results from each fold
        config: Training configuration
        base_timestamp: Base timestamp for folder naming
    """
    path = Path(os.getcwd())
    
    # Setup logging for results summary
    results_dir = Path(f'{path}/cross_validation_results/{base_timestamp}')
    logger = setup_cv_logging(results_dir)
    
    # Calculate mean and std for each metric
    fc_accuracies = [r['fc_metrics']['accuracy'] for r in cv_results]
    fc_f1_scores = [r['fc_metrics']['f1'] for r in cv_results]
    fc_precisions = [r['fc_metrics']['precision'] for r in cv_results]
    fc_recalls = [r['fc_metrics']['recall'] for r in cv_results]
    
    all_layers_accuracies = [r['all_layers_metrics']['accuracy'] for r in cv_results]
    all_layers_f1_scores = [r['all_layers_metrics']['f1'] for r in cv_results]
    all_layers_precisions = [r['all_layers_metrics']['precision'] for r in cv_results]
    all_layers_recalls = [r['all_layers_metrics']['recall'] for r in cv_results]
    
    # Aggregate results
    aggregated_results = {
        'config': config,
        'num_folds': len(cv_results),
        'fc_only_results': {
            'accuracy_mean': np.mean(fc_accuracies),
            'accuracy_std': np.std(fc_accuracies),
            'f1_mean': np.mean(fc_f1_scores),
            'f1_std': np.std(fc_f1_scores),
            'precision_mean': np.mean(fc_precisions),
            'precision_std': np.std(fc_precisions),
            'recall_mean': np.mean(fc_recalls),
            'recall_std': np.std(fc_recalls),
            'all_accuracies': fc_accuracies,
            'all_f1_scores': fc_f1_scores,
            'all_precisions': fc_precisions,
            'all_recalls': fc_recalls
        },
        'all_layers_results': {
            'accuracy_mean': np.mean(all_layers_accuracies),
            'accuracy_std': np.std(all_layers_accuracies),
            'f1_mean': np.mean(all_layers_f1_scores),
            'f1_std': np.std(all_layers_f1_scores),
            'precision_mean': np.mean(all_layers_precisions),
            'precision_std': np.std(all_layers_precisions),
            'recall_mean': np.mean(all_layers_recalls),
            'recall_std': np.std(all_layers_recalls),
            'all_accuracies': all_layers_accuracies,
            'all_f1_scores': all_layers_f1_scores,
            'all_precisions': all_layers_precisions,
            'all_recalls': all_layers_recalls
        },
        'fold_results': cv_results
    }
    
    # Create results directory
    results_dir = Path(f'{path}/cross_validation_results/{base_timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated results
    with open(results_dir / 'cv_results.json', 'w') as f:
        json.dump(aggregated_results, f, indent=4, default=str)
    
    # Create and save summary DataFrame
    summary_df = pd.DataFrame({
        'fold': [f"fold_{i+1}" for i in range(len(cv_results))] + ['mean', 'std'],
        'fc_accuracy': fc_accuracies + [np.mean(fc_accuracies), np.std(fc_accuracies)],
        'fc_f1': fc_f1_scores + [np.mean(fc_f1_scores), np.std(fc_f1_scores)],
        'fc_precision': fc_precisions + [np.mean(fc_precisions), np.std(fc_precisions)],
        'fc_recall': fc_recalls + [np.mean(fc_recalls), np.std(fc_recalls)],
        'all_layers_accuracy': all_layers_accuracies + [np.mean(all_layers_accuracies), np.std(all_layers_accuracies)],
        'all_layers_f1': all_layers_f1_scores + [np.mean(all_layers_f1_scores), np.std(all_layers_f1_scores)],
        'all_layers_precision': all_layers_precisions + [np.mean(all_layers_precisions), np.std(all_layers_precisions)],
        'all_layers_recall': all_layers_recalls + [np.mean(all_layers_recalls), np.std(all_layers_recalls)]
    })
    
    summary_df.to_csv(results_dir / 'cv_summary.csv', index=False)
    
    # Print summary
    logger.info("="*70)
    logger.info("CROSS-VALIDATION RESULTS SUMMARY")
    logger.info("="*70)
    logger.info(f"Configuration: ResNet34, {config['ris_option']}, {len(cv_results)}-fold CV")
    logger.info(f"\nFC-only training results:")
    logger.info(f"  Accuracy: {np.mean(fc_accuracies):.4f} ± {np.std(fc_accuracies):.4f}")
    logger.info(f"  F1 Score: {np.mean(fc_f1_scores):.4f} ± {np.std(fc_f1_scores):.4f}")
    logger.info(f"  Precision: {np.mean(fc_precisions):.4f} ± {np.std(fc_precisions):.4f}")
    logger.info(f"  Recall: {np.mean(fc_recalls):.4f} ± {np.std(fc_recalls):.4f}")
    
    logger.info(f"\nAll layers training results:")
    logger.info(f"  Accuracy: {np.mean(all_layers_accuracies):.4f} ± {np.std(all_layers_accuracies):.4f}")
    logger.info(f"  F1 Score: {np.mean(all_layers_f1_scores):.4f} ± {np.std(all_layers_f1_scores):.4f}")
    logger.info(f"  Precision: {np.mean(all_layers_precisions):.4f} ± {np.std(all_layers_precisions):.4f}")
    logger.info(f"  Recall: {np.mean(all_layers_recalls):.4f} ± {np.std(all_layers_recalls):.4f}")
    
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info("="*70)
    
    return aggregated_results


def run_cross_validation_resnet34(config: Optional[Dict] = None) -> Optional[Dict]:
    """
    Run k-fold cross validation with ResNet34 configuration.
    
    Args:
        config: Configuration dictionary with training parameters
    """
    # Set seed for reproducibility
    pl.seed_everything(config['random_seed'])

    # Create base timestamp
    base_timestamp = datetime.fromtimestamp(time.time()).strftime("%y-%m-%d_%H-%M-%S_ResNet34_CV")
    
    # Setup main CV logging
    path = Path(os.getcwd())
    cv_results_dir = Path(f'{path}/cross_validation_results/{base_timestamp}')
    logger = setup_cv_logging(cv_results_dir)

    logger.info(f"Starting {config['k_folds']}-fold cross-validation with ResNet34")
    logger.info(f"Configuration: {config}")
    
    # ULTRA-FAST TEST MODE: Monkey patch the data loading for extreme speed
    if config.get('test_run', False):
        logger.info("ULTRA-FAST TEST MODE: Sampling only 100 images per shot!")
        original_load_and_split = cmc.load_and_split_dataframes
        
        def fast_load_and_split_dataframes(path, shots, shots_for_training, shots_for_testing, 
                                         shots_for_validation, use_ELMS=True, ris_option='RIS1', 
                                         exponential_elm_decay=True):
            # Call original function
            shot_df, test_df, val_df, train_df = original_load_and_split(
                path, shots, shots_for_training, shots_for_testing, 
                shots_for_validation, use_ELMS, ris_option, exponential_elm_decay)
            
            # Sample only 100 rows per shot for ultra-fast testing
            def sample_shot_data(df, max_rows_per_shot=100):
                if len(df) == 0:
                    return df
                sampled_dfs = []
                for shot in df['shot'].unique():
                    shot_data = df[df['shot'] == shot]
                    if len(shot_data) > max_rows_per_shot:
                        shot_data = shot_data.sample(n=max_rows_per_shot, random_state=42)
                    sampled_dfs.append(shot_data)
                return pd.concat(sampled_dfs, ignore_index=True)
            
            # Apply sampling to all datasets
            shot_df = sample_shot_data(shot_df, 100)
            test_df = sample_shot_data(test_df, 100) 
            val_df = sample_shot_data(val_df, 100)
            train_df = sample_shot_data(train_df, 100)
            
            logger.info(f"ULTRA-FAST MODE: Reduced data to - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return shot_df, test_df, val_df, train_df
        
        # Apply monkey patch
        cmc.load_and_split_dataframes = fast_load_and_split_dataframes
    
    # Load shot data based on camera availability
    if config['ris_option'] == 'both':
        # For 'both' option, we need to get shots that have either RIS1 or RIS2 data
        shots_for_testing, shots_for_validation, shots_for_training = load_shot_data_for_both_cameras(
            path, config['test_df_contains_val_df'], config['test_run'], 
            config['data_frac'], config['random_seed'])
    else:
        # For single camera option, use the original function
        shots_for_testing, shots_for_validation, shots_for_training = LH.load_shot_data(
            path, config['ris_option'], config['test_df_contains_val_df'], config['test_run'], 
            config['data_frac'], config['random_seed'])

    # Combine training and validation shots for cross-validation
    all_train_val_shots = pd.concat([shots_for_training, shots_for_validation]).reset_index(drop=True)
    
    logger.info(f"Total shots for cross-validation: {len(all_train_val_shots)}")
    logger.info(f"Test shots (constant across folds): {len(shots_for_testing)}")
    
    # Create K-fold splits
    fold_splits = create_k_fold_splits(all_train_val_shots, config['k_folds'], config['random_seed'])
    
    # Run cross-validation
    cv_results = []
    
    for fold_idx, (train_shots, val_shots) in enumerate(fold_splits):
        try:
            fold_result = run_single_fold(
                fold_idx, train_shots, val_shots, shots_for_testing, 
                config, base_timestamp)
            cv_results.append(fold_result)
            
            # Clear GPU memory after each fold
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error in fold {fold_idx + 1}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Log the error to the run-specific directory
            cv_results_dir = Path(f'{path}/cross_validation_results/{base_timestamp}')
            error_log_path = cv_results_dir / f'cv_fold_{fold_idx + 1}_exception.log'
            error_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_log_path, 'w') as f:
                f.write(f"Fold {fold_idx + 1} error:\n")
                f.write(str(e) + '\n')
                f.write(traceback.format_exc())
            
            continue
    
    # Restore original function if we monkey patched it
    if config.get('test_run', False):
        cmc.load_and_split_dataframes = original_load_and_split
    
    if cv_results:
        # Aggregate and save results
        aggregated_results = aggregate_cv_results(cv_results, config, base_timestamp)
        return aggregated_results
    else:
        logger.error("No successful folds completed!")
        return None


if __name__ == '__main__':
    # Setup basic logging for main execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main_logger = logging.getLogger(__name__)


    # Configuration for cross-validation run with conservative settings
    config = {
        'ris_option': 'both',
        'num_workers': 4,  # Set to 0 to avoid multiprocessing issues in cross-validation
        'num_epochs_for_fc': 16,
        'num_epochs_for_all_layers': 16,
        'num_classes': 3,
        'batch_size': 32,  # Reduced batch size to avoid memory issues
        'learning_rate_min': 0.001,
        'learning_rate_max': 0.01,
        'weight_decay': 1e-3,
        'random_seed': 42,
        'augmentation': False,
        'test_df_contains_val_df': False,
        'test_run': False,
        'exponential_elm_decay': False,
        'grayscale': False,
        'data_frac': 1,     
        'k_folds': 5           
    }

    try:
        # Run 5-fold cross validation
        results = run_cross_validation_resnet34(config=config)
        
        if results:
            main_logger.info("Cross-validation completed successfully!")
        else:
            main_logger.error("Cross-validation failed!")
            
    except Exception as e:
        main_logger.error(f"Cross-validation error: {str(e)}")
        # Create a timestamp for the error log if one doesn't exist
        error_timestamp = datetime.fromtimestamp(time.time()).strftime("%y-%m-%d_%H-%M-%S_ResNet34_CV")
        error_log_path = Path(f'./cross_validation_results/{error_timestamp}/cv_exception.log')
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_path, 'w') as f:
            f.write(str(e) + '\n')
            f.write(traceback.format_exc())
