# USEF-TSE-Mamba v7 Project Structure

## Essential Files and Directories

```
USEF-TSE-Mamba-v7/
│
├── configs/                      # Configuration files
│   ├── config_v7.yaml           # Main training configuration
│   └── dataset_config_v7.yaml   # Dataset generation configuration
│
├── src/                         # Source code
│   ├── models/                  # Model architectures
│   │   ├── USEF_TSE_Mamba_v7.py         # Original v7 model
│   │   └── USEF_TSE_Mamba_v7_enhanced.py # Enhanced model with improvements
│   │
│   ├── data/                    # Data loading and processing
│   │   ├── generate_tse_dataset_v7.py   # Dataset generation script
│   │   ├── tse_dataset_v7.py            # Standard dataset loader
│   │   └── tse_dataset_v7_variable_ref.py # Variable reference dataset
│   │
│   ├── utils/                   # Utility functions
│   │   └── losses_v7.py         # Loss functions (SI-SDR, PIT)
│   │
│   ├── train_v7.py             # Main training script
│   ├── inference_v7.py         # Standard inference
│   └── inference_streaming.py  # Streaming inference for long audio
│
├── outputs/                     # Training outputs (created during training)
│   └── usef_tse_v7/            # Model outputs and logs
│
├── checkpoints/                 # Model checkpoints (created during training)
│   └── usef_tse_v7/            # Saved model weights
│
├── archive/                     # Archived non-essential files
│   ├── test_scripts/           # Test and validation scripts
│   ├── plots/                  # Generated plots and figures
│   ├── logs/                   # Old log files
│   └── reference_code/         # Reference implementations
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── PROJECT_STRUCTURE.md        # This file
```

## Key Components

### 1. Models (`src/models/`)
- **USEF_TSE_Mamba_v7.py**: Original implementation with cross-attention at all levels
- **USEF_TSE_Mamba_v7_enhanced.py**: Improved version with:
  - Batch normalization for stability
  - SiLU activation for better gradients
  - Efficient bottleneck-only attention
  - Gated skip connections

### 2. Data Pipeline (`src/data/`)
- **generate_tse_dataset_v7.py**: Creates TSE datasets from LibriSpeech + WHAM noise
- **tse_dataset_v7.py**: Standard PyTorch dataset with fixed reference lengths
- **tse_dataset_v7_variable_ref.py**: Advanced dataset with adaptive reference lengths

### 3. Training (`src/train_v7.py`)
- Mixed precision training support
- Gradient accumulation for larger effective batch sizes
- Warmup + ReduceLROnPlateau scheduler
- Comprehensive logging with TensorBoard

### 4. Inference Options
- **inference_v7.py**: Standard inference for evaluation
- **inference_streaming.py**: Process long audio files with sliding windows
  - Causal processing without overlap
  - Variable chunk sizes support
  - Real-time capable with low latency

## Configuration

### Training Configuration (`configs/config_v7.yaml`)
- Model architecture parameters
- Training hyperparameters
- Data loading settings
- Variable reference options

### Dataset Configuration (`configs/dataset_config_v7.yaml`)
- LibriSpeech paths
- WHAM noise paths
- Mixing parameters (SNR ranges)
- Output specifications

## Usage

### 1. Generate Dataset
```bash
python src/data/generate_tse_dataset_v7.py --config configs/dataset_config_v7.yaml
```

### 2. Train Model
```bash
python src/train_v7.py --config configs/config_v7.yaml
```

### 3. Run Inference
```bash
# Standard inference
python src/inference_v7.py \
    --model checkpoints/best_model.pth \
    --config configs/config_v7.yaml \
    --mode single \
    --mixture path/to/mixture.wav \
    --reference path/to/reference.wav

# Enhanced inference with post-processing
python src/inference_enhanced.py \
    --model checkpoints/best_model.pth \
    --config configs/config_v7.yaml \
    --mixture path/to/mixture.wav \
    --reference path/to/reference.wav \
    --output path/to/output.wav
```

## Model Selection

### Use Original Model for:
- Baseline comparisons
- Standard training scenarios
- When memory is very limited

### Use Enhanced Model for:
- Best performance (expected +1-2 dB SI-SDRi)
- Production deployments
- When training stability is important

## Data Options

### Fixed Reference (default):
- Consistent 3-second references
- Predictable memory usage
- Good for initial training

### Variable Reference (recommended):
- Adaptive lengths (2-3.5 seconds)
- Better generalization
- Expected +0.5-1 dB improvement

## Archive Contents

Non-essential files have been moved to `archive/`:
- Test scripts for development
- Generated plots and visualizations
- Log files from previous runs
- Reference implementations

These can be safely deleted if not needed for reference.