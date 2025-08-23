#!/usr/bin/env python3
"""
Script to find corrupted images in the dataset.
All images should be 640x500 (width x height) for the ResNet model.
Optimized version using PIL and multiprocessing for fast execution.
"""
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def check_image_batch(args):
    """Check a batch of images for one shot - designed for multiprocessing"""
    shot, csv_file, base_path, expected_dims = args
    
    EXPECTED_CHANNELS, EXPECTED_HEIGHT, EXPECTED_WIDTH = expected_dims
    
    corrupted_images = []
    missing_images = []
    total_checked = 0
    total_valid = 0
    
    try:
        if not os.path.exists(csv_file):
            return corrupted_images, missing_images, total_checked, total_valid, f"CSV not found for shot {shot}"
            
        df = pd.read_csv(csv_file)
        
        for idx, row in df.iterrows():
            img_filename = row['filename']
            img_path = Path(base_path) / img_filename
            total_checked += 1
            
            if not img_path.exists():
                missing_images.append({
                    'shot': shot,
                    'filename': img_filename,
                    'path': str(img_path),
                    'issue': 'FILE_NOT_FOUND'
                })
                continue
            
            try:
                # Use PIL to get image info without loading pixel data - much faster!
                with Image.open(img_path) as img:
                    width, height = img.size
                    # Get number of channels
                    if img.mode == 'RGB':
                        channels = 3
                    elif img.mode == 'RGBA':
                        channels = 4
                    elif img.mode == 'L':
                        channels = 1
                    else:
                        channels = len(img.getbands()) if hasattr(img, 'getbands') else 3
                    
                    # Check if dimensions are correct
                    if (channels != EXPECTED_CHANNELS or 
                        height != EXPECTED_HEIGHT or 
                        width != EXPECTED_WIDTH):
                        
                        corrupted_images.append({
                            'shot': shot,
                            'filename': img_filename,
                            'path': str(img_path),
                            'actual_size': f'{channels}x{height}x{width}',
                            'expected_size': f'{EXPECTED_CHANNELS}x{EXPECTED_HEIGHT}x{EXPECTED_WIDTH}',
                            'issue': 'WRONG_DIMENSIONS'
                        })
                    else:
                        total_valid += 1
                        
            except Exception as e:
                corrupted_images.append({
                    'shot': shot,
                    'filename': img_filename,
                    'path': str(img_path),
                    'error': str(e),
                    'issue': 'READ_ERROR'
                })
                
    except Exception as e:
        return corrupted_images, missing_images, total_checked, total_valid, f"Error processing shot {shot}: {e}"
    
    return corrupted_images, missing_images, total_checked, total_valid, None

def find_corrupted_images(max_workers=None, early_stop_count=None):
    """Find all corrupted images in the dataset using multiprocessing"""
    start_time = time.time()
    path = Path(os.getcwd())
    
    # Load shot usage data
    try:
        shot_usage = pd.read_csv(f'{path}/data/shot_usageNEW.csv')
        shot_for_ris1 = shot_usage[shot_usage['used_for_ris1']]
        shot_for_ris2 = shot_usage[shot_usage['used_for_ris2']]
        
        all_shots = set(shot_for_ris1['shot'].values) | set(shot_for_ris2['shot'].values)
        print(f"Found {len(all_shots)} shots to check")
    except Exception as e:
        print(f"Error loading shot_usage: {e}")
        return
    
    # Expected image dimensions for ResNet model
    EXPECTED_CHANNELS = 3
    EXPECTED_HEIGHT = 500
    EXPECTED_WIDTH = 640
    expected_dims = (EXPECTED_CHANNELS, EXPECTED_HEIGHT, EXPECTED_WIDTH)
    
    # Prepare arguments for multiprocessing
    tasks = []
    for shot in sorted(all_shots):
        csv_file = f'{path}/data/LH_alpha/LH_alpha_shot_{shot}.csv'
        tasks.append((shot, csv_file, path, expected_dims))
    
    # Use multiprocessing
    if max_workers is None:
        max_workers = min(cpu_count(), 8)  # Cap at 8 to avoid overwhelming I/O
    
    print(f"Using {max_workers} worker processes")
    
    corrupted_images = []
    missing_images = []
    total_images_checked = 0
    total_valid_images = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_shot = {executor.submit(check_image_batch, task): task[0] for task in tasks}
        
        # Process results as they complete
        with tqdm(total=len(tasks), desc="Processing shots") as pbar:
            for future in as_completed(future_to_shot):
                shot = future_to_shot[future]
                try:
                    batch_corrupted, batch_missing, batch_checked, batch_valid, error = future.result()
                    
                    if error:
                        print(f"Warning: {error}")
                        continue
                    
                    corrupted_images.extend(batch_corrupted)
                    missing_images.extend(batch_missing)
                    total_images_checked += batch_checked
                    total_valid_images += batch_valid
                    
                    # Early stopping if requested
                    if early_stop_count and len(corrupted_images) >= early_stop_count:
                        print(f"\nEarly stopping: Found {len(corrupted_images)} corrupted images")
                        # Cancel remaining futures
                        for remaining_future in future_to_shot:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                        
                except Exception as exc:
                    print(f"Shot {shot} generated an exception: {exc}")
                finally:
                    pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Report results
    print(f"\n{'='*60}")
    print("CORRUPTED IMAGE DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Processing speed: {total_images_checked/elapsed_time:.0f} images/second")
    print(f"Total images checked: {total_images_checked}")
    print(f"Valid images: {total_valid_images}")
    print(f"Corrupted/wrong size images: {len(corrupted_images)}")
    print(f"Missing images: {len(missing_images)}")
    
    if corrupted_images:
        print(f"\nCORRUPTED IMAGES ({len(corrupted_images)} found):")
        print("-" * 80)
        
        # Group by issue type
        issues_by_type = {}
        for img in corrupted_images:
            issue_type = img['issue']
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(img)
        
        for issue_type, images in issues_by_type.items():
            print(f"\n{issue_type} ({len(images)} images):")
            for img in images[:10]:  # Show first 10 of each type
                if 'actual_size' in img:
                    print(f"  Shot {img['shot']}: {img['filename']} - Size: {img['actual_size']} (expected: {img['expected_size']})")
                elif 'error' in img:
                    print(f"  Shot {img['shot']}: {img['filename']} - Error: {img['error']}")
                else:
                    print(f"  Shot {img['shot']}: {img['filename']}")
            
            if len(images) > 10:
                print(f"  ... and {len(images) - 10} more")
    
    if missing_images:
        print(f"\nMISSING IMAGES ({len(missing_images)} found):")
        print("-" * 80)
        for img in missing_images[:20]:  # Show first 20
            print(f"  Shot {img['shot']}: {img['filename']}")
        if len(missing_images) > 20:
            print(f"  ... and {len(missing_images) - 20} more")
    
    # Save detailed report to file
    if corrupted_images or missing_images:
        report_file = path / 'corrupted_images_report.txt'
        with open(report_file, 'w') as f:
            f.write("CORRUPTED IMAGE DETECTION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total images checked: {total_images_checked}\n")
            f.write(f"Valid images: {total_valid_images}\n")
            f.write(f"Corrupted/wrong size images: {len(corrupted_images)}\n")
            f.write(f"Missing images: {len(missing_images)}\n\n")
            
            if corrupted_images:
                f.write("CORRUPTED IMAGES:\n")
                f.write("-" * 40 + "\n")
                for img in corrupted_images:
                    f.write(f"Shot {img['shot']}: {img['filename']}\n")
                    f.write(f"  Path: {img['path']}\n")
                    if 'actual_size' in img:
                        f.write(f"  Size: {img['actual_size']} (expected: {img['expected_size']})\n")
                    if 'error' in img:
                        f.write(f"  Error: {img['error']}\n")
                    f.write(f"  Issue: {img['issue']}\n\n")
            
            if missing_images:
                f.write("\nMISSING IMAGES:\n")
                f.write("-" * 40 + "\n")
                for img in missing_images:
                    f.write(f"Shot {img['shot']}: {img['filename']}\n")
                    f.write(f"  Path: {img['path']}\n\n")
        
        print(f"\nDetailed report saved to: {report_file}")
    
    # Generate cleanup script
    if corrupted_images:
        cleanup_script = path / 'cleanup_corrupted_images.sh'
        with open(cleanup_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Script to remove corrupted images\n")
            f.write("# Review carefully before running!\n\n")
            for img in corrupted_images:
                f.write(f'# Shot {img["shot"]}: {img["issue"]}\n')
                f.write(f'# rm "{img["path"]}"\n\n')
        
        print(f"Cleanup script template saved to: {cleanup_script}")
        print("Review the cleanup script before running it!")
    
    if not corrupted_images and not missing_images:
        print("\n✅ No corrupted or missing images found! All images are valid.")
    
    return corrupted_images, missing_images

def find_corrupted_images_fast_sample(sample_count=1000):
    """Quick sampling mode - check only a subset of images for fast feedback"""
    print(f"Fast sampling mode: checking {sample_count} random images")
    
    path = Path(os.getcwd())
    
    # Load shot usage data
    try:
        shot_usage = pd.read_csv(f'{path}/data/shot_usageNEW.csv')
        shot_for_ris1 = shot_usage[shot_usage['used_for_ris1']]
        shot_for_ris2 = shot_usage[shot_usage['used_for_ris2']]
        
        all_shots = list(set(shot_for_ris1['shot'].values) | set(shot_for_ris2['shot'].values))
        print(f"Sampling from {len(all_shots)} available shots")
    except Exception as e:
        print(f"Error loading shot_usage: {e}")
        return
    
    import random
    random.shuffle(all_shots)
    
    EXPECTED_CHANNELS = 3
    EXPECTED_HEIGHT = 500
    EXPECTED_WIDTH = 640
    
    corrupted_count = 0
    checked_count = 0
    
    for shot in all_shots:
        if checked_count >= sample_count:
            break
            
        csv_file = f'{path}/data/LH_alpha/LH_alpha_shot_{shot}.csv'
        if not os.path.exists(csv_file):
            continue
            
        try:
            df = pd.read_csv(csv_file)
            for idx, row in df.iterrows():
                if checked_count >= sample_count:
                    break
                    
                img_filename = row['filename']
                img_path = Path(path) / img_filename
                checked_count += 1
                
                if not img_path.exists():
                    print(f"Missing: {img_filename}")
                    continue
                
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        channels = 3 if img.mode == 'RGB' else len(img.getbands())
                        
                        if (channels != EXPECTED_CHANNELS or 
                            height != EXPECTED_HEIGHT or 
                            width != EXPECTED_WIDTH):
                            print(f"Wrong size: {img_filename} - {channels}x{height}x{width}")
                            corrupted_count += 1
                            
                except Exception as e:
                    print(f"Read error: {img_filename} - {e}")
                    corrupted_count += 1
                    
        except Exception as e:
            continue
    
    print(f"\nSample results: {corrupted_count}/{checked_count} images have issues")
    if corrupted_count > 0:
        estimated_total = (corrupted_count / checked_count) * 300000
        print(f"Estimated total corrupted images: ~{estimated_total:.0f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        # Fast sampling mode
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        find_corrupted_images_fast_sample(sample_size)
    elif len(sys.argv) > 1 and sys.argv[1] == "fast":
        # Fast mode with early stopping
        early_stop = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        find_corrupted_images(early_stop_count=early_stop)
    else:
        # Full mode (optimized)
        find_corrupted_images()
