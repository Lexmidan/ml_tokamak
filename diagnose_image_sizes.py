#!/usr/bin/env python3
"""
Diagnostic script to check image sizes in the dataset to identify corrupted/mismatched images.
"""
import os
import pandas as pd
from pathlib import Path
from torchvision.io import read_image
from collections import Counter
import traceback

def check_image_sizes():
    """Check sizes of all images in the dataset"""
    path = Path(os.getcwd())
    
    # Load shot usage data
    try:
        shot_usage = pd.read_csv(f'{path}/data/shot_usageNEW.csv')
        shot_for_ris = shot_usage[shot_usage['used_for_ris2']]
        shots = shot_for_ris['shot'].values
        print(f"Found {len(shots)} shots in shot_usageNEW.csv")
    except Exception as e:
        print(f"Error loading shot_usage: {e}")
        return
    
    # Check some data files to understand the structure
    try:
        for shot in shots[:5]:  # Just check first 5 shots
            csv_file = f'{path}/data/LH_alpha/LH_alpha_shot_{shot}.csv'
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                print(f"Shot {shot}: {len(df)} entries")
                if 'filename' in df.columns:
                    print(f"  Sample filenames: {df['filename'].head(3).tolist()}")
                break
    except Exception as e:
        print(f"Error loading sample data: {e}")
    
    # Collect image sizes
    image_sizes = []
    problematic_images = []
    
    # Check first few shots for analysis
    shots_to_check = shots[:10] if len(shots) > 10 else shots
    
    for shot in shots_to_check:
        csv_file = f'{path}/data/LH_alpha/LH_alpha_shot_{shot}.csv'
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found for shot {shot}")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            # Sample some images from this shot
            sample_size = min(20, len(df))  # Sample at most 20 images per shot
            sampled_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
            
            for idx, row in sampled_df.iterrows():
                try:
                    img_path = Path(path) / row['filename']
                    if img_path.exists():
                        image = read_image(str(img_path))
                        size = tuple(image.shape)  # (channels, height, width)
                        image_sizes.append(size)
                        
                        # Check if this is an unusual size
                        if size[1] != 500 or size[2] != 640:  # Expected 640x500
                            problematic_images.append({
                                'shot': shot,
                                'filename': row['filename'],
                                'size': size,
                                'path': str(img_path)
                            })
                    else:
                        print(f"Image not found: {img_path}")
                        
                except Exception as e:
                    print(f"Error reading image {row['filename']}: {e}")
                    problematic_images.append({
                        'shot': shot,
                        'filename': row['filename'],
                        'error': str(e),
                        'path': str(Path(path) / row['filename'])
                    })
                    
        except Exception as e:
            print(f"Error processing shot {shot}: {e}")
    
    # Analyze results
    print(f"\nAnalyzed {len(image_sizes)} images from {len(shots_to_check)} shots")
    
    if image_sizes:
        size_counter = Counter(image_sizes)
        print(f"\nImage size distribution:")
        for size, count in size_counter.most_common():
            print(f"  {size}: {count} images")
        
        if problematic_images:
            print(f"\nFound {len(problematic_images)} problematic images:")
            for img in problematic_images[:10]:  # Show first 10
                if 'error' in img:
                    print(f"  ERROR - Shot {img['shot']}: {img['filename']} - {img['error']}")
                else:
                    print(f"  WRONG SIZE - Shot {img['shot']}: {img['filename']} - Size: {img['size']}")
            
            if len(problematic_images) > 10:
                print(f"  ... and {len(problematic_images) - 10} more")
        else:
            print("No problematic images found in the sample!")
    else:
        print("No valid images found!")

if __name__ == "__main__":
    check_image_sizes()
