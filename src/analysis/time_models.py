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

# Add the project root directory to sys.path (relative to this file's location)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
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
    """Simple dataset for timing classification"""
    def __init__(self, image_paths, mean, std, grayscale=False):
        self.image_paths = image_paths
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.grayscale = grayscale
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path).float()
        
        if self.grayscale:
            # Convert to grayscale using luminance weights
            weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(3, 1, 1)
            image = (image * weights).sum(dim=0, keepdim=True)
            mean = torch.tensor([0.485], dtype=torch.float32).view(1, 1, 1)  # Approximate grayscale mean
            std = torch.tensor([0.229], dtype=torch.float32).view(1, 1, 1)   # Approximate grayscale std
        else:
            mean = self.mean[:, None, None]
            std = self.std[:, None, None]
        
        # Normalize
        normalized_image = (image - mean) / (255 * std).float()
        
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

def time_model_classification(model_name='resnet18', n_images=1000, batch_sizes=[1, 8, 16, 32], 
                            grayscale=False, ris_option='RIS1', warmup_batches=10, 
                            print_func=print, logger=None):
    """
    Time classification duration for ResNet models
    
    Args:
        model_name: 'resnet18' or 'resnet34'
        n_images: Number of images to test on
        batch_sizes: List of batch sizes to test
        grayscale: Whether to use grayscale images
        ris_option: 'RIS1', 'RIS2', or 'both'
        warmup_batches: Number of warmup batches to run before timing
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
    
    # Create dataset
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    dataset = TimingDataset(image_paths, mean, std, grayscale)
    
    results = {}
    
    # Progress bar for batch sizes
    batch_pbar = tqdm(batch_sizes, desc=f"Testing {model_name}", leave=False)
    
    for batch_size in batch_pbar:
        batch_start_time = time.time()
        batch_pbar.set_description(f"Testing {model_name} - batch size {batch_size}")
        
        print_func(f"\nTesting batch size: {batch_size}")
        if logger:
            logger.info(f"Starting batch size {batch_size} testing")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Warmup
        print_func("Warming up...")
        if logger:
            logger.info(f"Starting warmup with {warmup_batches} batches")
        
        with torch.no_grad():
            warmup_pbar = tqdm(range(warmup_batches), desc="Warmup", leave=False, disable=(logger is None))
            warmup_count = 0
            for batch in dataloader:
                if warmup_count >= warmup_batches:
                    break
                batch = batch.to(device)
                _ = model(batch)
                warmup_count += 1
                warmup_pbar.update(1)
            warmup_pbar.close()
        
        # Timing
        print_func("Starting timing...")
        if logger:
            logger.info("Starting actual timing measurements")
        
        batch_times = []
        total_images_processed = 0
        
        with torch.no_grad():
            # Progress bar for timing batches
            timing_pbar = tqdm(dataloader, desc=f"Timing batch_size={batch_size}", leave=False)
            
            for batch in timing_pbar:
                batch = batch.to(device)
                
                # Time this batch
                torch.cuda.synchronize() if device.type == 'cuda' else None
                batch_start = time.time()
                
                _ = model(batch)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                batch_end = time.time()
                
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)
                total_images_processed += batch.size(0)
                
                # Update progress bar with current FPS
                current_fps = batch.size(0) / batch_time
                timing_pbar.set_postfix({'FPS': f'{current_fps:.1f}'})
            
            timing_pbar.close()
        
        # Calculate statistics
        batch_times = np.array(batch_times)
        total_time = np.sum(batch_times)
        
        # Time per image statistics
        time_per_image = total_time / total_images_processed
        
        # Also calculate per-image times for each batch to get std
        per_image_times = batch_times / np.array([min(batch_size, len(dataset) - i*batch_size) 
                                                for i in range(len(batch_times))])
        
        results[batch_size] = {
            'total_time': total_time,
            'total_images': total_images_processed,
            'mean_time_per_image': time_per_image,
            'std_time_per_image': np.std(per_image_times),
            'mean_batch_time': np.mean(batch_times),
            'std_batch_time': np.std(batch_times),
            'throughput_fps': total_images_processed / total_time
        }
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        
        result_info = (f"  Processed {total_images_processed} images in {total_time:.4f} seconds "
                      f"(batch test took {batch_duration:.1f}s)")
        timing_info = f"  Mean time per image: {time_per_image*1000:.2f} ± {np.std(per_image_times)*1000:.2f} ms"
        throughput_info = f"  Throughput: {total_images_processed/total_time:.1f} FPS"
        
        print_func(result_info)
        print_func(timing_info)
        print_func(throughput_info)
        
        if logger:
            logger.info(f"Batch size {batch_size} results:")
            logger.info(result_info)
            logger.info(timing_info)
            logger.info(throughput_info)
    
    batch_pbar.close()
    
    total_duration = time.time() - start_time
    completion_info = f"Model {model_name} testing completed in {total_duration:.1f} seconds"
    print_func(f"\n{completion_info}")
    if logger:
        logger.info(completion_info)
    
    return results

# Main execution
def run_timing_benchmark(save_to_file=True, output_file=None, log_file=None):
    """Run timing benchmark for both ResNet18 and ResNet34"""
    
    benchmark_start_time = time.time()
    
    # Setup logging
    logger, actual_log_file = setup_logging(log_file)
    logger.info("="*80)
    logger.info("STARTING TIMING BENCHMARK")
    logger.info("="*80)
    
    models_to_test = ['resnet18', 'resnet34']
    batch_sizes = [1, 8, 16, 32, 64]  # Test different batch sizes
    n_images = 1000
    
    logger.info(f"Configuration:")
    logger.info(f"  Models to test: {models_to_test}")
    logger.info(f"  Batch sizes: {batch_sizes}")
    logger.info(f"  Number of images: {n_images}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup output file
    if save_to_file and output_file is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_file = f"timing_results_{timestamp}.txt"
    
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
        rgb_results = time_model_classification(
            model_name=model_name,
            n_images=n_images,
            batch_sizes=batch_sizes,
            grayscale=False,
            ris_option='RIS1',
            print_func=print_both,
            logger=logger
        )
        overall_pbar.update(1)
        
        # Test with grayscale images
        print_both(f"\nTesting {model_name} with grayscale images:")
        logger.info(f"Testing {model_name} with grayscale images")
        gray_results = time_model_classification(
            model_name=model_name,
            n_images=n_images,
            batch_sizes=batch_sizes,
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
    print_both("SUMMARY RESULTS")
    print_both(f"{'='*80}")
    
    logger.info("Generating summary results")
    
    for model_name in models_to_test:
        print_both(f"\n{model_name.upper()} Results:")
        print_both("-" * 40)
        
        for color_mode in ['rgb', 'grayscale']:
            print_both(f"\n{color_mode.upper()} Images:")
            results = all_results[model_name][color_mode]
            
            for batch_size, stats in results.items():
                mean_ms = stats['mean_time_per_image'] * 1000
                std_ms = stats['std_time_per_image'] * 1000
                fps = stats['throughput_fps']
                
                result_line = (f"  Batch size {batch_size:2d}: "
                             f"{mean_ms:6.2f} ± {std_ms:5.2f} ms/image "
                             f"({fps:6.1f} FPS)")
                print_both(result_line)
    
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

# Run the benchmark
# Save to file by default with timestamp and comprehensive logging
results = run_timing_benchmark(save_to_file=True)

# Alternative usage examples:
# Custom filenames:
# results = run_timing_benchmark(save_to_file=True, output_file="my_timing_results.txt", log_file="my_timing.log")

# Console only (original behavior):
# results = run_timing_benchmark(save_to_file=False)

# Quick test with specific log file:
# results = run_timing_benchmark(save_to_file=True, log_file="quick_test.log")
