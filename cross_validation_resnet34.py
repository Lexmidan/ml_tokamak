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

import LHmode_classifier as LH
import confinement_mode_classifier as cmc


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
    
    # Create dataloaders for this fold
    dataloaders, dataset_sizes, test_dataloader = LH.create_dataloaders(
        path, train_shots, test_shots, val_shots,
        config['ris_option'], config['num_classes'], config['exponential_elm_decay'], 
        config['batch_size'], config['num_workers'], config['augmentation'], config['grayscale'])

    # Setup model
    pretrained_model = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model = LH.setup_model(pretrained_model, config['num_classes'], device, config['grayscale'])

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
    
    # Load shot data
    shots_for_testing, shots_for_validation, shots_for_training = LH.load_shot_data(
        path, 'RIS1' if config['ris_option'] == 'both' else config['ris_option'], 
        config['test_df_contains_val_df'], config['test_run'], 
        config['data_frac'], config['random_seed'])
    
    # Handle 'both' option by combining RIS1 and RIS2 data
    if config['ris_option'] == 'both':
        shots_for_testing_ris2, shots_for_validation_ris2, shots_for_training_ris2 = LH.load_shot_data(
            path, 'RIS2', config['test_df_contains_val_df'], config['test_run'], 
            config['data_frac'], config['random_seed'])
        
        shots_for_testing = pd.concat([shots_for_testing, shots_for_testing_ris2]).reset_index(drop=True)
        shots_for_validation = pd.concat([shots_for_validation, shots_for_validation_ris2]).reset_index(drop=True)
        shots_for_training = pd.concat([shots_for_training, shots_for_training_ris2]).reset_index(drop=True)

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


    # Configuration for ultra-fast test run 
    config = {
        'ris_option': 'both',
        'num_workers': 4,
        'num_epochs_for_fc': 16,
        'num_epochs_for_all_layers': 16,
        'num_classes': 3,
        'batch_size': 32,
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
