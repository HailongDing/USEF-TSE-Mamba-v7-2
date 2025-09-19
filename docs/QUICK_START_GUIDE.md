# USEF-TSE-Mamba v7 - Complete Quick Start Guide

This guide walks you through the entire process from dataset preparation to training and inference.

## Prerequisites

### 1. System Requirements
- GPU: NVIDIA RTX 3090/4090 or better (24GB VRAM)
- RAM: 32GB or more recommended
- Storage: ~500GB for datasets and checkpoints
- OS: Linux (Ubuntu 20.04+ recommended)

### 2. Environment Setup

```bash
# Create conda environment
conda create -n usef_tse python=3.10
conda activate usef_tse

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## Step 1: Prepare Datasets

### 1.1 Download LibriSpeech Dataset

```bash
# Create data directory
mkdir -p /data/LibriSpeech

# Download LibriSpeech (clean versions)
cd /data/LibriSpeech

# Training data (~55GB)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz

# Validation data (~1.4GB)
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/dev-other.tar.gz

# Test data (~700MB)
wget https://www.openslr.org/resources/12/test-clean.tar.gz

# Extract all archives
for file in *.tar.gz; do tar -xzf $file; done
```

### 1.2 Download WHAM! Noise Dataset

```bash
# Create noise directory
mkdir -p /data/wham_noise

# Download WHAM noise (~17GB)
cd /data/wham_noise

# Download from: https://wham.whisper.ai/
# You need to register and download:
# - wham_noise.zip

# Extract
unzip wham_noise.zip
```

### 1.3 Verify Dataset Structure

```bash
# LibriSpeech structure should be:
/data/LibriSpeech/
├── train-clean-100/
├── train-clean-360/
├── dev-clean/
├── dev-other/
└── test-clean/

# WHAM structure should be:
/data/wham_noise/
├── tr/  # Training noise
├── cv/  # Validation noise
└── tt/  # Test noise
```

## Step 2: Generate TSE Dataset

### 2.1 Configure Dataset Generation

Edit `configs/dataset_config_v7.yaml` to verify paths:

```yaml
paths:
  librispeech_root: /data/LibriSpeech
  wham_noise_root: /data/wham_noise
  output_root: /data/tse_audio_dataset_v7

dataset_sizes:
  train: 20000      # Number of training mixtures
  validation: 5000  # Number of validation mixtures
  test: 3000        # Number of test mixtures

mixing:
  snr_range: [0, 10]        # SNR between speakers
  noise_snr_range: [0, 10]  # Background noise SNR
```

### 2.2 Generate the Dataset

```bash
cd /home/ubuntu/latest/USEF-TSE-Mamba-v7

# Activate environment
conda activate usef_tse

# Generate dataset (this will take 2-4 hours)
python src/data/generate_tse_dataset_v7.py --config configs/dataset_config_v7.yaml

# Expected output:
# Generating train split: 20000 mixtures
# Generating validation split: 5000 mixtures
# Generating test split: 3000 mixtures
# Dataset generation complete!
```

### 2.3 Verify Generated Dataset

```bash
# Check generated files
ls -la /data/tse_audio_dataset_v7/

# Should see:
# train/
# validation/
# test/

# Check file count
ls /data/tse_audio_dataset_v7/train/*.wav | wc -l
# Should show 60000 files (20000 × 3 files per sample)
```

## Step 3: Configure Training

### 3.1 Choose Model Configuration

Edit `configs/config_v7.yaml`:

```yaml
# Model selection
model:
  dim: 128
  kernel_sizes: [16, 16, 16]
  strides: [2, 2, 2]
  num_blocks: 6
  activation: silu      # Use 'silu' for enhanced model
  num_heads: 8
  dropout: 0.1

# Data configuration
data:
  data_root: /data/tse_audio_dataset_v7
  sample_rate: 8000
  segment_length: 5.0   # 5-second segments
  batch_size: 1         # Adjust based on GPU memory

  # Variable reference (recommended)
  use_variable_ref: true
  ref_length_strategy: adaptive
  memory_safe_mode: true

# Training configuration
training:
  num_epochs: 150
  mixed_precision: true
  gradient_accumulation: 4  # Effective batch size = 4

  # Learning rate warmup
  warmup_epochs: 5
  warmup_start_lr: 1.0e-5

# Optimizer
optimizer:
  type: adamw
  learning_rate: 0.0001
  weight_decay: 0.01

# Scheduler
scheduler:
  type: reduce_on_plateau
  factor: 0.5
  patience: 10
  min_lr: 1.0e-6

# Paths
paths:
  output_dir: ./outputs/usef_tse_v7
  checkpoint_dir: ./checkpoints/usef_tse_v7
```

### 3.2 Choose Model Implementation

For best performance, modify `src/train_v7.py` to use enhanced model:

```python
# Line 15-16, change from:
from models.USEF_TSE_Mamba_v7 import USEF_TSE_Mamba_v7

# To:
from models.USEF_TSE_Mamba_v7_enhanced import USEF_TSE_Mamba_v7_Enhanced as USEF_TSE_Mamba_v7
```

## Step 4: Train the Model

### 4.1 Start Training

```bash
# Single GPU training
python src/train_v7.py --config configs/config_v7.yaml

# Expected output:
# Loading data...
# Train: 20000 samples, 20000 batches
# Val: 5000 samples, 5000 batches
# Starting training on cuda
# Epoch 1/150: loss=15.23, SI-SDRi=0.12 dB
# ...
```

### 4.2 Monitor Training

```bash
# In another terminal, monitor with TensorBoard
tensorboard --logdir outputs/usef_tse_v7/runs

# Open browser to http://localhost:6006
```

### 4.3 Resume Training (if interrupted)

```bash
# Find latest checkpoint
ls -lt checkpoints/usef_tse_v7/*.pth

# Resume from checkpoint
python src/train_v7.py \
    --config configs/config_v7.yaml \
    --resume checkpoints/usef_tse_v7/checkpoint_epoch_50.pth
```

## Step 5: Inference

### 5.1 Prepare Test Audio

```bash
# Create test directory
mkdir -p test_samples

# Copy some test files from generated dataset
cp /data/tse_audio_dataset_v7/test/mixture_0001.wav test_samples/
cp /data/tse_audio_dataset_v7/test/reference_0001.wav test_samples/
cp /data/tse_audio_dataset_v7/test/target_0001.wav test_samples/
```

### 5.2 Run Standard Inference

```bash
# Single file inference
python src/inference_v7.py \
    --model checkpoints/usef_tse_v7/best_model.pth \
    --config configs/config_v7.yaml \
    --mode single \
    --mixture test_samples/mixture_0001.wav \
    --reference test_samples/reference_0001.wav \
    --target test_samples/target_0001.wav \
    --output output/extracted_0001.wav

# Expected output:
# Model loaded from checkpoint
# Processing: test_samples/mixture_0001.wav
# SI-SDRi: 10.52 dB
# Output saved to: output/extracted_0001.wav
```

### 5.3 Run Streaming Inference (for long audio)

```bash
# Streaming inference for long audio files
python src/inference_streaming.py

# The script uses default test audio files or you can modify paths in the script
# Expected output:
# USEF-TSE-Mamba v7 - Streaming Inference (Causal)
# Processing 36.0s audio in chunks...
# Completed: 36.0s audio in 1.05s (RTF: 0.03x)
# Output saved to: test_audio/extracted_streaming.wav
```

### 5.4 Batch Inference

```bash
# Process entire test set
python src/inference_v7.py \
    --model checkpoints/usef_tse_v7/best_model.pth \
    --config configs/config_v7.yaml \
    --mode batch \
    --input_dir /data/tse_audio_dataset_v7/test \
    --output_dir output/test_results

# Expected output:
# Processing 3000 files...
# Average SI-SDRi: 10.8 ± 2.1 dB
# Results saved to: output/test_results/
```

## Step 6: Evaluate Performance

### 6.1 Run Full Evaluation

```bash
python src/inference_v7.py \
    --model checkpoints/usef_tse_v7/best_model.pth \
    --config configs/config_v7.yaml \
    --mode evaluate \
    --test_dir /data/tse_audio_dataset_v7/test \
    --num_samples 1000

# Expected output:
# Evaluation Results:
# SI-SDRi: 10.85 ± 2.13 dB
# Processing time: 0.23s per sample
```

## Expected Timeline

| Stage | Estimated Time |
|-------|---------------|
| Environment Setup | 30 minutes |
| Dataset Download | 1-2 hours (depends on internet) |
| Dataset Generation | 2-4 hours |
| Training (150 epochs) | 24-48 hours |
| Inference (per file) | < 1 second |

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory (OOM) during training**
   ```yaml
   # Reduce batch_size in config
   batch_size: 1
   # Increase gradient_accumulation
   gradient_accumulation: 8
   ```

2. **Dataset generation fails**
   ```bash
   # Check disk space
   df -h /data
   # Need ~200GB free space
   ```

3. **Slow training**
   ```yaml
   # Enable mixed precision
   mixed_precision: true
   # Reduce segment_length
   segment_length: 3.0
   ```

4. **Poor performance**
   ```yaml
   # Use enhanced model
   # Increase training epochs
   num_epochs: 200
   # Use variable references
   use_variable_ref: true
   ```

## Performance Expectations

With proper training, you should achieve:

| Checkpoint | Expected SI-SDRi |
|------------|-----------------|
| Epoch 10 | 3-4 dB |
| Epoch 30 | 6-7 dB |
| Epoch 50 | 8-9 dB |
| Epoch 100 | 9-10 dB |
| Epoch 150 | 10-11 dB |
| Best (enhanced) | 11-12 dB |

## Next Steps

After successful training:

1. **Fine-tuning**: Train with longer segments (6-8 seconds) for better performance
2. **Domain Adaptation**: Fine-tune on your specific audio domain
3. **Deployment**: Use streaming inference for real-time applications
4. **Optimization**: Convert to ONNX for faster inference

## Support

If you encounter issues:
1. Check the logs in `outputs/usef_tse_v7/`
2. Verify GPU memory with `nvidia-smi`
3. Ensure all paths in configs are correct
4. Check dataset generation logs in `archive/logs/`

Good luck with your training!