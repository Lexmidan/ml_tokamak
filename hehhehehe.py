import os
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm

# Constants
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 500
IMG_ROOT = Path('./imgs')
OUTPUT_CSV = 'corrupted_images.csv'

def check_image(img_path):
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
                return {
                    'shot': img_path.parent.name,
                    'filename': str(img_path),
                    'actual_size': f'{width}x{height}'
                }
    except Exception as e:
        return {
            'shot': img_path.parent.name,
            'filename': str(img_path),
            'actual_size': f'ERROR: {e}'
        }
    return None

def collect_all_image_paths(root_dir):
    return list(root_dir.rglob("*.png"))

def main():
    print("Collecting image paths...")
    all_images = collect_all_image_paths(IMG_ROOT)
    print(f"Found {len(all_images)} images to check")

    num_workers = min(cpu_count(), 8)  # Limit to avoid I/O overload
    print(f"Using {num_workers} parallel workers")

    corrupted = []

    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(check_image, all_images, chunksize=100), total=len(all_images)):
            if result:
                corrupted.append(result)

    # Save results
    if corrupted:
        df = pd.DataFrame(corrupted)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n❌ Found {len(corrupted)} corrupted images. Saved to '{OUTPUT_CSV}'")
    else:
        print("\n✅ All images have correct dimensions.")

if __name__ == "__main__":
    main()