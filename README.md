# COMPASS Tokamak Confinement Mode Classification

Machine learning framework for classifying plasma confinement modes (L-mode, H-mode, ELM) in the COMPASS Tokamak using various neural network architectures and data sources.


### Model Architectures

- **ResNet-based Models** (ResNet18/34/50) - Single image classification from RIS1/RIS2 fast cameras
- **PhyDNet** - Physics-informed neural network for sequence-based classification [2]
- **InceptionTime** - Advanced time-series classifier for 1D signal data (Mirnov coils, Langmuir probes, H_α) [1]
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
```bash
# Using the main training script
python train.py --model resnet --ris RIS1 --epochs-fc 10 --epochs-all 20

# Or programmatically
python -c "
from src.utils.confinement_mode_classifier import train_and_test_ris_model
model, model_path = train_and_test_ris_model(
    ris_option='RIS1',
    num_epochs_for_fc=10,
    num_epochs_for_all_layers=20
)
"
```

#### 2. **Physics-informed Sequence Classification (PhyDNet)**
```bash
# Using the main training script
python train.py --model phydnet

# Or directly
python src/training/PhyDNet_COMPASS.py --num_epochs 50 --batch_size 16
```

#### 3. **1D Signal Classification**
```bash
# Using the main training script
python train.py --model alt_models

# Or programmatically
python -c "
from src.training.alt_models_training import train_and_test_alt_model
train_and_test_alt_model()
"
```

#### 4. **Data Processing**
```bash
# Process image data
python process_data.py --data-type images

# Process signal data
python process_data.py --data-type signals --shot-numbers 16534 16535 16536
```

#### 5. **Cross-Validation**
```bash
python cross_validate.py
```

## Project Structure

The project is now organized into a modular structure for better maintainability:

### Entry Scripts (Root Level)
- `train.py` - Main training script with unified CLI interface
- `process_data.py` - Data processing entry point
- `cross_validate.py` - Cross-validation runner

### Core Modules (`src/`)

#### Training (`src/training/`)
- `LHmode_classifier.py` - ResNet-based image classifiers
- `alt_models_training.py` - 1D signal classification models
- `PhyDNet_COMPASS.py` - Physics-informed sequence models
- `cross_validation_resnet34.py` - Cross-validation implementation
- `PhyDNet_finetuning.py` - PhyDNet fine-tuning utilities

#### Data Processing (`src/data_processing/`)
- `imgs_processing.py` - Image dataset preparation and preprocessing
- `process_data_for_alt_models.py` - 1D signal data preparation

#### Model Definitions (`src/models/`)
- `alt_models.py` - InceptionTime and Simple1DCNN implementations
- `PhyDNet_models.py` - PhyDNet architecture components

#### Analysis & Visualization (`src/analysis/`)
- `visual.py` - Visualization utilities
- `results_visualization.ipynb` - Interactive results exploration
- `notebooks/` - Jupyter notebooks for analysis and testing

#### Utilities (`src/utils/`)
- `confinement_mode_classifier.py` - Core utilities for ResNet and 1D signal models
- `find_corrupted_images.py` - Dataset integrity checking
- `change_permissions.py` - File permission management
- `ModelEnsembling.py` - NN ensemble implementation (**Deprecated**)

### Examples (`examples/`)
- `refactored_examples.py` - Usage examples and tutorials
- `routine.py` - Example training routines

## Complete Workflow

### Step 1: Data Preparation
```bash
# For image-based models
python process_data.py --data-type images

# For 1D signal models  
python process_data.py --data-type signals --shot-numbers 16534 16535 16536
```

### Step 2: Model Training

**Single Image Classification:**
```bash
# Using the unified training script
python train.py --model resnet --ris RIS1 --batch-size 32

# Or programmatically
python -c "
from src.utils.confinement_mode_classifier import train_and_test_ris_model
model, path = train_and_test_ris_model(
    ris_option='both',  # Use images from both cameras
    model_name='resnet34',
    batch_size=32,
    grayscale=True
)
"
```

**Sequence-based Classification:**
```bash
# Using the unified training script
python train.py --model phydnet

# Or directly
python src/training/PhyDNet_COMPASS.py --sequence_length 10 --batch_size 8
```

**1D Signal Classification:**
```bash
# Using the unified training script
python train.py --model alt_models

# Or directly configure and run
python src/training/alt_models_training.py
```

### Step 3: Results Analysis
```bash
# Launch interactive visualization
jupyter notebook src/analysis/results_visualization.ipynb
```

## Model Performance & Cross-Validation

Run k-fold cross-validation:
```bash
# Using the cross-validation script
python cross_validate.py

# Or programmatically
python -c "
from src.training.cross_validation_resnet34 import run_cross_validation
results = run_cross_validation(
    model_name='resnet34',
    ris_option='RIS1', 
    n_splits=5
)
"
```

View training progress:
```bash
tensorboard --logdir=./runs
```

## Advanced Usage

### Custom Model Architecture
```python
from src.utils.confinement_mode_classifier import create_model_from_config, train_and_test_ris_model

# Create custom ResNet variant
model = create_model_from_config('resnet50')
trained_model, path = train_and_test_ris_model(pretrained_model=model)
```

### Hyperparameter Tuning
See `examples/refactored_examples.py` for detailed examples of:
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

