import time
import os
import sys
from pathlib import Path
import random
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from src.training.LHmode_classifier import create_model_from_config, setup_model


def setup_logging(log_file=None):
    """Setup logging configuration"""
    if log_file is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"timing_benchmark_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger('timing_benchmark')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger, log_file


class TimingDataset(Dataset):
    """Optimized dataset for timing classification - minimizes per-sample overhead"""
    def __init__(self, image_paths, mean, std, grayscale=False):
        self.image_paths = image_paths
        self.grayscale = grayscale
        
        # Pre-compute normalization tensors to avoid creating them every time
        if grayscale:
            self.mean = torch.tensor([0.485], dtype=torch.float32).view(1, 1, 1)
            self.std = torch.tensor([0.229], dtype=torch.float32).view(1, 1, 1)
            self.rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(3, 1, 1)
        else:
            self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path).float()
        
        if self.grayscale:
            # Convert to grayscale using pre-computed weights
            image = (image * self.rgb_weights).sum(dim=0, keepdim=True)
        
        # Normalize using pre-computed tensors
        normalized_image = (image - self.mean) / (255 * self.std)
        
        return normalized_image

def get_random_image_paths(data_path, n_images=1000, ris_option='RIS1', logger=None):
    """Get random image paths from the dataset"""
    data_path = Path(data_path)
    
    if logger:
        logger.info(f"Loading shot usage data from {data_path / 'data/shot_usageNEW.csv'}")
    
    # Load shot usage data
    shot_usage = pd.read_csv(data_path / 'data/shot_usageNEW.csv')
    
    # Get shots used for the specified RIS option
    if ris_option == 'RIS1':
        available_shots = shot_usage[shot_usage['used_for_ris1']]['shot'].tolist()
    else:
        available_shots = shot_usage[shot_usage['used_for_ris2']]['shot'].tolist()
    
    
    if logger:
        logger.info(f"Found {len(available_shots)} available shots for {ris_option}")
    
    # Collect all image paths
    all_image_paths = []
    
    # Add progress bar for shot processing
    shot_pbar = tqdm(available_shots, desc="Processing shots", disable=(logger is None))
    
    for shot in shot_pbar:
        try:
            # Load the CSV file for this shot
            shot_df = pd.read_csv(data_path / f'data/LH_alpha/LH_alpha_shot_{shot}.csv')
            
            # Get image paths
            for filename in shot_df['filename']:
                img_path = data_path / filename
                if img_path.exists():
                    all_image_paths.append(str(img_path))
                    
                # Also add RIS2 images if ris_option includes it
                if ris_option in ['RIS2', 'both']:
                    ris2_filename = filename.replace('RIS1', 'RIS2')
                    ris2_img_path = data_path / 'data/LH_alpha' / ris2_filename
                    if ris2_img_path.exists():
                        all_image_paths.append(str(ris2_img_path))
                        
        except FileNotFoundError:
            if logger:
                logger.warning(f"Could not find data for shot {shot}")
            continue
    
    shot_pbar.close()
    
    if logger:
        logger.info(f"Collected {len(all_image_paths)} total image paths")
    
    # Randomly sample n_images
    if len(all_image_paths) < n_images:
        if logger:
            logger.warning(f"Only found {len(all_image_paths)} images, using all available")
        return all_image_paths
    
    selected_paths = random.sample(all_image_paths, n_images)
    if logger:
        logger.info(f"Randomly selected {len(selected_paths)} images for timing")
    
    return selected_paths

def time_real_time_inference(model_name='resnet18', n_images=1000, grayscale=False, 
                           ris_option='RIS1', warmup_samples=20, print_func=print, logger=None):
    """
    Time single-image inference for real-time applications
    
    Args:
        model_name: 'resnet18' or 'resnet34'
        n_images: Number of images to test on
        grayscale: Whether to use grayscale images
        ris_option: 'RIS1', 'RIS2', or 'both'
        warmup_samples: Number of warmup inferences to run before timing
        print_func: Print function to use (for logging to file)
        logger: Logger instance for detailed logging
    """
    
    start_time = time.time()
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_info = f"Using device: {device}"
    if device.type == 'cuda':
        device_info += f" ({torch.cuda.get_device_name(device)})"
    
    print_func(device_info)
    if logger:
        logger.info(device_info)
    
    # Create model
    model_info = f"Loading {model_name} model ({'grayscale' if grayscale else 'RGB'})..."
    print_func(model_info)
    if logger:
        logger.info(model_info)
    
    model = create_model_from_config(model_name)
    model = setup_model(model, num_classes=3, device=device, grayscale=grayscale)
    model.eval()
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_info = f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
    print_func(param_info)
    if logger:
        logger.info(param_info)
    
    # Get data path
    data_path = Path(r'/compass/Shared/Users/bogdanov/ml_tokamak/')
    
    # Get random image paths
    print_func(f"Collecting {n_images} random image paths...")
    if logger:
        logger.info(f"Starting image collection for {n_images} images from {ris_option}")
    
    image_paths = get_random_image_paths(data_path, n_images, ris_option, logger)
    
    collection_info = f"Found {len(image_paths)} images"
    print_func(collection_info)
    if logger:
        logger.info(collection_info)
    
    # Create dataset with optimized parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    dataset = TimingDataset(image_paths, mean, std, grayscale)
    
    # Pre-load samples to GPU memory for pure inference timing
    print_func("Pre-loading samples to GPU memory...")
    if logger:
        logger.info("Pre-loading samples to GPU memory for pure inference timing")
    
    # Load a subset for warmup and timing
    test_samples = min(n_images, len(dataset))
    samples = []
    
    # Use progress bar for sample loading
    sample_pbar = tqdm(range(test_samples), desc="Loading samples", disable=(logger is None))
    for i in sample_pbar:
        sample = dataset[i].unsqueeze(0).to(device)
        samples.append(sample)
    sample_pbar.close()
    
    print_func(f"Loaded {len(samples)} samples to GPU memory")
    if logger:
        logger.info(f"Loaded {len(samples)} samples to GPU memory")
    
    # Warmup phase
    print_func(f"Warming up with {warmup_samples} samples...")
    if logger:
        logger.info(f"Starting warmup with {warmup_samples} samples")
    
    with torch.no_grad():
        warmup_pbar = tqdm(range(warmup_samples), desc="Warmup", leave=False, disable=(logger is None))
        for i in warmup_pbar:
            sample_idx = i % len(samples)
            _ = model(samples[sample_idx])
        warmup_pbar.close()
    
    # Single-image inference timing (real-time scenario)
    print_func("Starting single-image inference timing...")
    if logger:
        logger.info("Starting single-image inference timing for real-time scenario")
    
    inference_times = []
    
    with torch.no_grad():
        timing_pbar = tqdm(samples, desc="Timing single images", leave=False)
        
        for sample in timing_pbar:
            # Time pure inference (no data loading)
            torch.cuda.synchronize()
            inference_start = time.time()
            
            output = model(sample)
            
            torch.cuda.synchronize()
            inference_end = time.time()
            
            inference_time = inference_end - inference_start
            inference_times.append(inference_time)
            
            # Update progress bar with current timing
            current_ms = inference_time * 1000
            timing_pbar.set_postfix({'ms': f'{current_ms:.2f}'})
        
        timing_pbar.close()
    
    # Calculate comprehensive statistics
    inference_times = np.array(inference_times)
    
    # Basic statistics
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    median_time = np.median(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    # Percentiles for real-time analysis
    p95_time = np.percentile(inference_times, 95)
    p99_time = np.percentile(inference_times, 99)
    
    # Convert to milliseconds
    mean_ms = mean_time * 1000
    std_ms = std_time * 1000
    median_ms = median_time * 1000
    min_ms = min_time * 1000
    max_ms = max_time * 1000
    p95_ms = p95_time * 1000
    p99_ms = p99_time * 1000
    
    # Real-time performance metrics
    max_fps = 1.0 / mean_time
    guaranteed_fps_95 = 1.0 / p95_time  # 95% of inferences will be faster than this
    guaranteed_fps_99 = 1.0 / p99_time  # 99% of inferences will be faster than this
    
    results = {
        'total_samples': len(inference_times),
        'mean_time_ms': mean_ms,
        'std_time_ms': std_ms,
        'median_time_ms': median_ms,
        'min_time_ms': min_ms,
        'max_time_ms': max_ms,
        'p95_time_ms': p95_ms,
        'p99_time_ms': p99_ms,
        'max_fps': max_fps,
        'guaranteed_fps_95': guaranteed_fps_95,
        'guaranteed_fps_99': guaranteed_fps_99,
        'raw_times': inference_times
    }
    
    # Print results
    print_func(f"\nREAL-TIME INFERENCE RESULTS:")
    print_func(f"  Samples tested: {len(inference_times)}")
    print_func(f"  Mean inference time: {mean_ms:.2f} ± {std_ms:.2f} ms")
    print_func(f"  Median inference time: {median_ms:.2f} ms")
    print_func(f"  Min/Max: {min_ms:.2f} / {max_ms:.2f} ms")
    print_func(f"  95th percentile: {p95_ms:.2f} ms")
    print_func(f"  99th percentile: {p99_ms:.2f} ms")
    print_func(f"")
    print_func(f"  Real-time performance:")
    print_func(f"    Maximum FPS: {max_fps:.1f}")
    print_func(f"    Guaranteed FPS (95%): {guaranteed_fps_95:.1f}")
    print_func(f"    Guaranteed FPS (99%): {guaranteed_fps_99:.1f}")
    

    if logger:
        logger.info("Real-time inference timing results:")
        logger.info(f"Mean: {mean_ms:.2f} ± {std_ms:.2f} ms")
        logger.info(f"95th percentile: {p95_ms:.2f} ms")
        logger.info(f"99th percentile: {p99_ms:.2f} ms")
        logger.info(f"Guaranteed FPS (99%): {guaranteed_fps_99:.1f}")
    
    total_duration = time.time() - start_time
    completion_info = f"Model {model_name} real-time testing completed in {total_duration:.1f} seconds"
    print_func(f"\n{completion_info}")
    if logger:
        logger.info(completion_info)
    
    return results

# Main execution
def run_timing_benchmark(save_to_file=True, output_file=None, log_file=None):
    """Run timing benchmark for both ResNet18 and ResNet34 with focus on real-time single-image inference"""
    
    benchmark_start_time = time.time()
    
    # Setup logging
    logger, actual_log_file = setup_logging(log_file)
    logger.info("="*80)
    logger.info("STARTING REAL-TIME INFERENCE BENCHMARK")
    logger.info("="*80)
    
    models_to_test = ['resnet18', 'resnet34']
    n_images = 1000
    
    logger.info(f"Configuration:")
    logger.info(f"  Models to test: {models_to_test}")
    logger.info(f"  Test type: Single-image real-time inference")
    logger.info(f"  Number of images: {n_images}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup output file
    if save_to_file and output_file is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f"timing_results_{timestamp}.log"
    
    # Redirect print function if saving to file
    if save_to_file:
        results_file = open(output_file, 'w')
        def print_both(*args, **kwargs):
            print(*args, **kwargs)  # Print to console
            print(*args, **kwargs, file=results_file)  # Print to file
            results_file.flush()  # Ensure immediate write
    else:
        print_both = print
    
    all_results = {}
    
    # Overall progress bar
    total_tests = len(models_to_test) * 2  # Each model tested with RGB and grayscale
    overall_pbar = tqdm(total=total_tests, desc="Overall Progress")
    
    for model_idx, model_name in enumerate(models_to_test):
        model_start_time = time.time()
        
        print_both(f"\n{'='*60}")
        print_both(f"TESTING {model_name.upper()} ({model_idx+1}/{len(models_to_test)})")
        print_both(f"{'='*60}")
        
        logger.info(f"Starting {model_name} testing")
        
        # Test with RGB images
        print_both(f"\nTesting {model_name} with RGB images:")
        logger.info(f"Testing {model_name} with RGB images")
        rgb_results = time_real_time_inference(
            model_name=model_name,
            n_images=n_images,
            grayscale=False,
            ris_option='RIS1',
            print_func=print_both,
            logger=logger
        )
        overall_pbar.update(1)
        
        # Test with grayscale images
        print_both(f"\nTesting {model_name} with grayscale images:")
        logger.info(f"Testing {model_name} with grayscale images")
        gray_results = time_real_time_inference(
            model_name=model_name,
            n_images=n_images,
            grayscale=True,
            ris_option='RIS1',
            print_func=print_both,
            logger=logger
        )
        overall_pbar.update(1)
        
        all_results[model_name] = {
            'rgb': rgb_results,
            'grayscale': gray_results
        }
        
        model_duration = time.time() - model_start_time
        logger.info(f"Completed {model_name} testing in {model_duration:.1f} seconds")
    
    overall_pbar.close()
    
    # Print summary
    print_both(f"\n{'='*80}")
    print_both("REAL-TIME INFERENCE SUMMARY")
    print_both(f"{'='*80}")
    
    logger.info("Generating summary results")
    
    for model_name in models_to_test:
        print_both(f"\n{model_name.upper()} Real-Time Performance:")
        print_both("-" * 50)
        
        for color_mode in ['rgb', 'grayscale']:
            print_both(f"\n{color_mode.upper()} Images:")
            results = all_results[model_name][color_mode]
            
            mean_ms = results['mean_time_ms']
            std_ms = results['std_time_ms']
            p99_ms = results['p99_time_ms']
            max_fps = results['max_fps']
            guaranteed_fps_99 = results['guaranteed_fps_99']
            
            print_both(f"  Mean inference time: {mean_ms:.2f} ± {std_ms:.2f} ms")
            print_both(f"  99th percentile: {p99_ms:.2f} ms")
            print_both(f"  Maximum FPS: {max_fps:.1f}")
            print_both(f"  Guaranteed FPS (99%): {guaranteed_fps_99:.1f}")
    
    
    # Final timing and cleanup
    total_duration = time.time() - benchmark_start_time
    completion_msg = f"\nBenchmark completed in {total_duration/60:.1f} minutes ({total_duration:.1f} seconds)"
    print_both(completion_msg)
    logger.info(completion_msg)
    
    # Close files if we opened them
    if save_to_file:
        file_info = f"\nResults saved to: {output_file}"
        log_info = f"Detailed logs saved to: {actual_log_file}"
        print_both(file_info)
        print_both(log_info)
        logger.info(file_info)
        logger.info(log_info)
        results_file.close()
    
    logger.info("="*80)
    logger.info("BENCHMARK COMPLETED")
    logger.info("="*80)
    
    return all_results

results = run_timing_benchmark(save_to_file=True)
