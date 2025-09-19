# USEF-TSE-Mamba V7 (Optimized)

Universal Speaker Embedding-Free Target Speaker Extraction with State Space Models

## 🚀 Quick Start

### Installation & Training

```bash
# 1. Activate environment
conda activate usef_tse

# 2. Generate optimized dataset (30K/2K/1K split)
./regenerate_dataset_optimized.sh

# 3. Train model with optimized configuration
python src/train_v7.py --config configs/config_v7_optimized.yaml
```

## 📁 Clean Project Structure

```
USEF-TSE-Mamba-v7/
├── src/                              # Core source code
│   ├── models/
│   │   └── USEF_TSE_Mamba_v7_enhanced.py  # Enhanced model (14.2GB memory)
│   ├── data/
│   │   ├── generate_tse_dataset_v7_parallel.py  # Fast parallel generation
│   │   ├── tse_dataset_v7_cached.py            # LRU cached loader
│   │   └── tse_dataset_v7_augmented.py         # Augmentation pipeline
│   ├── utils/
│   │   ├── losses_v7.py                        # SI-SDR loss
│   │   └── ema.py                              # EMA for stability
│   ├── train_v7.py                             # Main training script
│   └── inference_v7.py                         # Inference script
│
├── configs/
│   ├── config_v7_optimized.yaml               # Training config (30K/2K/1K)
│   └── dataset_config_v7_optimized.yaml       # Dataset generation config
│
├── scripts/
│   └── regenerate_dataset_optimized.sh        # Dataset generation script
│
└── docs/                                       # Documentation
    ├── QUICK_START_GUIDE.md
    ├── ANTI_OVERFITTING_GUIDE.md
    └── EXTENDED_AUDIO_CONFIG.md
```

## 🎯 Key Improvements in V7

### Model Architecture (Enhanced)
- **Bottleneck-only attention**: 58% memory reduction (23.5GB → 14.2GB)
- **Anti-overfitting**: Dropout 0.2, EMA, weight decay 0.05
- **Better activations**: SiLU + BatchNorm + gated skips
- **Target**: 11-13 dB SI-SDRi (vs 8.75 dB in v6)

### Dataset Configuration (Optimized)
- **Training**: 30,000 samples (50 hours) - 50% more data
- **Validation**: 2,000 samples (3.3 hours) - 60% less overhead
- **Test**: 1,000 samples (1.7 hours) - Sufficient for metrics
- **Audio**: 6s mixtures, 4s references @ 8kHz

### Training Optimizations
- **Smart validation**: Quick (500) every 5 epochs, full every 10
- **Parallel data loading**: 16 workers with prefetch
- **LRU cache**: 1024 file cache for repeated access
- **Mixed precision**: Automatic mixed precision training
- **Augmentation**: MixUp, SpecAugment, speed perturbation

## 📊 Expected Performance

| Metric | V6 Baseline | V7 Optimized | Improvement |
|--------|-------------|--------------|-------------|
| SI-SDRi | 8.75 dB | 11-13 dB | +2-4 dB |
| Train/Val Gap | 4.6 dB | < 2 dB | Less overfitting |
| Memory Usage | 23.5 GB | 14.2 GB | -40% |
| Training Time | 36h | 47h | More data |

## 🛠️ Key Configuration

```yaml
# Model (config_v7_optimized.yaml)
model:
  dropout: 0.2
  use_bottleneck_attention_only: true

# Training
training:
  validation_frequency: 5      # Quick val every 5 epochs
  full_val_frequency: 10      # Full val every 10 epochs
  use_augmentation: true
  use_ema: true

# Data
data:
  segment_length: 6.0         # Extended context
  ref_length_max: 4.0         # Richer speaker features
  num_workers: 16             # Parallel loading
```

## 📈 Monitoring

```bash
# Watch training progress
tensorboard --logdir outputs/usef_tse_v7_optimized

# Check GPU memory
watch -n 1 nvidia-smi
```

## 📚 Documentation

- [Quick Start Guide](docs/QUICK_START_GUIDE.md) - Detailed setup
- [Anti-Overfitting Guide](docs/ANTI_OVERFITTING_GUIDE.md) - Regularization strategies
- [Extended Audio Config](docs/EXTENDED_AUDIO_CONFIG.md) - Audio length analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@software{usef_tse_mamba_v7,
  title = {USEF-TSE-Mamba v7: Universal Speaker Embedding Free Target Speaker Extraction},
  year = {2024},
  version = {6.0}
}
```

## License

This project is for research purposes. Please check individual component licenses.