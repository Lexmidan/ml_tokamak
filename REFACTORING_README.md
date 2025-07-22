# LHmode_classifier Refactoring

This document describes the refactoring of the `LHmode_classifier.py` file to make it more modular, reusable, and maintainable.

## What Was Changed

### Before: Single Monolithic Function
- One huge `train_and_test_ris_model()` function (~300 lines)
- Hard to reuse components
- Difficult to test individual parts
- Code duplication for similar operations
- Not suitable for cross-validation or hyperparameter sweeps

### After: Modular Architecture
The code has been broken down into focused, reusable functions:

## Core Functions

### Data Loading and Preparation
- **`load_shot_data()`**: Load and split shot data for training/validation/testing
- **`create_dataloaders()`**: Create PyTorch dataloaders with proper configurations

### Model Setup and Configuration
- **`setup_model()`**: Configure pretrained model for transfer learning
- **`prepare_model_for_grayscale()`**: Modify model for grayscale input
- **`create_model_from_config()`**: Create models from string configurations
- **`create_optimizer_and_scheduler()`**: Setup training components

### Training and Testing
- **`train_phase()`**: Train model for a specific phase (FC-only or all layers)
- **`test_and_save_results()`**: Test model and save predictions/metrics
- **`train_single_fold()`**: Train a single cross-validation fold

### Utilities
- **`save_hyperparameters_and_metrics()`**: Save experiment results to JSON/TensorBoard

## Benefits of Refactoring

### 1. **Reusability**
- Functions can be used independently
- Easy to create cross-validation pipelines
- Simple to implement hyperparameter sweeps
- Components can be mixed and matched

### 2. **Maintainability**
- Each function has a single responsibility
- Easier to debug and modify
- Clear function signatures with type hints
- Comprehensive docstrings

### 3. **Testability**
- Individual components can be unit tested
- Easier to isolate and fix bugs
- Reproducible experiments

### 4. **Flexibility**
- Easy to experiment with different model architectures
- Simple to modify training procedures
- Configurable for different datasets (RIS1, RIS2, both)

## Usage Examples

### Basic Training (Same as Before)
```python
from LHmode_classifier import train_and_test_ris_model

model, model_path = train_and_test_ris_model()
```

### Custom Model Architecture
```python
from LHmode_classifier import train_and_test_ris_model, create_model_from_config

custom_model = create_model_from_config('resnet34')
model, model_path = train_and_test_ris_model(
    pretrained_model=custom_model,
    grayscale=True,
    batch_size=64
)
```

### Cross-Validation
```python
from cross_validation_utils import run_cross_validation

cv_results = run_cross_validation(
    ris_option='RIS1',
    model_name='resnet18',
    n_splits=5,
    num_epochs_for_fc=10,
    num_epochs_for_all_layers=10
)
```

### Manual Pipeline Control
```python
from LHmode_classifier import load_shot_data, create_dataloaders, setup_model, train_phase

# Load data
shots_testing, shots_validation, shots_training = load_shot_data(
    path, 'RIS1', test_run=True)

# Create dataloaders
dataloaders, dataset_sizes, test_dataloader = create_dataloaders(
    path, shots_training, shots_testing, shots_validation, 'RIS1', ...)

# Setup and train model
model = setup_model(pretrained_model, num_classes=3, device=device)
model = train_phase(model, dataloaders, dataset_sizes, ...)
```

## New Capabilities

### Cross-Validation Support
The `cross_validation_utils.py` module provides:
- K-fold cross-validation
- Model comparison utilities
- Statistical analysis of results

### Hyperparameter Sweeps
Easy to implement systematic hyperparameter exploration:
```python
for lr in [0.001, 0.01]:
    for batch_size in [16, 32, 64]:
        model, path = train_and_test_ris_model(
            learning_rate_max=lr,
            batch_size=batch_size,
            comment_for_model_name=f'_lr{lr}_bs{batch_size}'
        )
```

### Model Architecture Comparison
```python
from cross_validation_utils import compare_models

results = compare_models(
    model_names=['resnet18', 'resnet34', 'resnet50'],
    ris_option='RIS1',
    n_splits=5
)
```

## File Structure

```
├── LHmode_classifier.py          # Main refactored training functions
├── cross_validation_utils.py     # Cross-validation utilities
├── refactored_examples.py        # Usage examples
└── REFACTORING_README.md         # This documentation
```

## Migration Guide

### For Existing Scripts
The main function signature remains the same, so existing code should work without changes:
```python
# This still works exactly as before
model, model_path = train_and_test_ris_model()
```

### For New Development
Use the modular functions for better control:
```python
# More flexible and reusable approach
shots = load_shot_data(path, 'RIS1')
dataloaders, sizes, test_dl = create_dataloaders(path, *shots, 'RIS1', ...)
model = setup_model(pretrained_model, num_classes=3, device=device)
```

## Type Hints and Documentation

All functions now include:
- **Type hints** for parameters and return values
- **Comprehensive docstrings** explaining purpose and usage
- **Clear parameter descriptions**
- **Example usage patterns**

This makes the code more self-documenting and easier to use with IDEs and static analysis tools.
