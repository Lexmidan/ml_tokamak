# COMPASS Tokamak Confinement Mode Classification

A comprehensive machine learning framework for classifying plasma confinement modes (L-mode, H-mode, ELM) in the COMPASS Tokamak using various neural network architectures.

## Project Overview

This project implements multiple deep learning approaches to classify plasma confinement modes in COMPASS Tokamak data:

### Model Architectures

- **ResNet-based Models** (ResNet18/34/50) - Single image classification from RIS1/RIS2 fast cameras
- **PhyDNet** - Physics-informed neural network for sequence-based classification
- **InceptionTime** - Advanced time-series classifier for 1D signal data (Mirnov coils, Langmuir probes, H_α)
- **Simple1DCNN** - Lightweight convolutional network for 1D signals

### Data Sources

- **RIS1/RIS2 Fast Cameras**: High-speed visible light imaging
- **Mirnov Coils**: Magnetic fluctuation measurements  
- **Langmuir Probes**: Plasma edge diagnostics
- **H_α Spectroscopy**: Hydrogen emission monitoring

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

**Note**: This project requires access to COMPASS Tokamak data through CDBClient (internal module).

### Basic Usage

#### 1. **Image-based Classification (ResNet)**
```python
from confinement_mode_classifier import train_and_test_ris_model

# Train on RIS1 camera data
model, model_path = train_and_test_ris_model(
    ris_option='RIS1',
    num_epochs_for_fc=10,
    num_epochs_for_all_layers=20
)
```

#### 2. **Physics-informed Sequence Classification (PhyDNet)**
```python
# Train PhyDNet model
python PhyDNet_COMPASS.py --num_epochs 50 --batch_size 16
```

#### 3. **1D Signal Classification**
```python
from alt_models_training import main

# Train InceptionTime or Simple1DCNN
main()
```

## Project Structure

### Core Training Scripts
- `LHmode_classifier.py` - ResNet-based image classifiers
- `alt_models_training.py` - 1D signal classification models
- `PhyDNet_COMPASS.py` - Physics-informed sequence models  
- `confinement_mode_classifier.py` - utilities for ResNet and 1D signal models
- `cross_validation_resnet34.py` - Cross-validation script
- `ModelEnsembling.py` - NN ensemble of two ResNet receiving either two images from different cameras or different times. **Deprecated** (single model is enough)

### Data Processing
- `imgs_processing.py` - Image dataset preparation and preprocessing
- `process_data_for_alt_models.py` - 1D signal data preparation

### Model Definitions
- `alt_models.py` - InceptionTime and Simple1DCNN implementations
- `PhyDNet_models.py` - PhyDNet architecture components

### Analysis & Visualization
- `results_visualization.ipynb` - Interactive results exploration
- `visual.py` - Visualization utilities
- `notebooks/` - Jupyter notebooks for analysis and testing

## 🔄 Complete Workflow

### Step 1: Data Preparation
```bash
# For image-based models
python imgs_processing.py

# For 1D signal models  
python process_data_for_alt_models.py
```

### Step 2: Model Training

**Single Image Classification:**
```python
from confinement_mode_classifier import train_and_test_ris_model

# Basic training
model, path = train_and_test_ris_model(ris_option='RIS1')

# Custom configuration
model, path = train_and_test_ris_model(
    ris_option='both',  # Use images from both cameras
    model_name='resnet34',
    batch_size=32,
    grayscale=True
)
```

**Sequence-based Classification:**
```python
# PhyDNet training
python PhyDNet_COMPASS.py --sequence_length 10 --batch_size 8
```

**1D Signal Classification:**
```python
# Configure in alt_models_training.py, then run:
python alt_models_training.py
```

### Step 3: Results Analysis
```python
# Launch interactive visualization
jupyter notebook results_visualization.ipynb
```

## Model Performance & Cross-Validation

Run k-fold cross-validation:
```python
from cross_validation_resnet34 import run_cross_validation

results = run_cross_validation(
    model_name='resnet34',
    ris_option='RIS1', 
    n_splits=5
)
```

View training progress:
```bash
tensorboard --logdir=./runs
```

## Advanced Usage

### Custom Model Architecture
```python
from confinement_mode_classifier import create_model_from_config

# Create custom ResNet variant
model = create_model_from_config('resnet50')
trained_model, path = train_and_test_ris_model(pretrained_model=model)
```

### Hyperparameter Tuning
See `refactored_examples.py` for detailed examples of:
- Custom training configurations
- Model ensemble methods
- Manual training pipeline control

## Output Structure

Training results are saved in `./runs/` with timestamps:
```
runs/
├── YYYY-MM-DD_HH-MM-SS_ModelName_RISOption/
│   ├── model.pth                 # Trained model weights
│   ├── hyperparameters.json      # Training configuration  
│   ├── metrics.json              # Performance metrics
│   ├── predictions.csv           # Test set predictions
│   └── tensorboard_logs/         # TensorBoard event files
```

## Physics Background

This project addresses plasma confinement mode classification in COMPASS tokamak, specifically targeting:

- **L-mode**: Low confinement mode with continuous turbulent transport
- **H-mode**: High confinement mode with improved particle/energy confinement  
- **ELM**: Edge Localized Modes - periodic instabilities in H-mode. Different types of ELMs are not considered. Neither is Dithering.

## Citation & References

[1] M. Zorek, et al.: Semi-supervised deep networks for plasma state identification. Plasma Phys. Control. Fusion 64 (2022) 125004.

[2] V. Le Guen, et al.: Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction. CVPR (2020).

[3] V. Weinzettl, et al.: Progress in diagnostics of the COMPASS tokamak. JINST 12 (2017) C12015.

