# Test Run Optimization Summary

## Original Issue
You correctly identified that `test_run=True` and `data_frac=0.3` were working, but the code was still loading a lot of data. This is because:

1. **`test_run=True`** limits to 3 shots per dataset (train/val/test)
2. **`data_frac=0.3`** samples 30% of training shots
3. **However**: Each shot contains ~2000 images (time series data points)

So even with 3 shots, you're loading ~6000 images per dataset!

## Optimizations Applied

### 1. Original Working Parameters
- `test_run=True` → Limits to 3 shots per dataset 
- `data_frac=0.3` → Uses 30% of training data

### 2. First-Level Optimizations
```python
config = {
    'ris_option': 'RIS1',      # Was 'both' - reduces data by 50%
    'num_workers': 2,          # Was 4 - reduces memory usage
    'batch_size': 16,          # Was 32 - smaller batches
    'data_frac': 0.1,          # Was 0.3 - only 10% of training data
    'k_folds': 2               # Was 5 - faster cross-validation
}
```

### 3. Second-Level Optimizations
```python
config = {
    'ris_option': 'RIS1',      # Single camera only
    'num_workers': 2,
    'batch_size': 8,           # Even smaller batches
    'grayscale': True,         # Reduces memory by 3x (vs RGB)
    'data_frac': 0.05,         # Only 5% of training data
    'k_folds': 2
}
```

### 4. Ultra-Fast Mode (Monkey Patch)
Added automatic data sampling when `test_run=True`:
- Samples only **100 images per shot** (vs ~2000 originally)
- Reduces data size by **95%** per shot
- Applied to all datasets (train/val/test)

## Expected Data Reduction

| Configuration | Images per Shot | Total Shots | Total Images | Reduction |
|---------------|-----------------|-------------|--------------|-----------|
| **Original**  | ~2000          | 3 × 3 = 9   | ~18,000     | Baseline |
| **Level 1**   | ~2000          | 3 × 3 = 9   | ~9,000      | 50% (RIS1 only) |
| **Level 2**   | ~2000          | 3 × 3 = 9   | ~9,000      | 50% + grayscale |
| **Ultra-Fast**| **100**        | 3 × 3 = 9   | **~900**    | **95%** |

## Memory Benefits
- **Grayscale**: 3x less memory (1 channel vs 3 RGB channels)
- **Smaller batches**: Fits in limited GPU memory
- **Fewer workers**: Reduces CPU/memory overhead
- **Image sampling**: 20x less data to load and process

## How to Run Ultra-Fast Test

The current configuration will automatically:
1. Use only RIS1 camera data
2. Sample 5% of available training shots
3. Use grayscale images  
4. **Automatically sample only 100 images per shot** when `test_run=True`
5. Run only 2-fold cross-validation

Just run the script as normal - the optimizations are automatic when `test_run=True`.

## Verification

You can verify the optimizations are working by checking the logs:
- Look for: `"ULTRA-FAST TEST MODE: Sampling only 100 images per shot!"`
- Look for: `"ULTRA-FAST MODE: Reduced data to - Train: X, Val: Y, Test: Z"`

The numbers should be much smaller than before!
