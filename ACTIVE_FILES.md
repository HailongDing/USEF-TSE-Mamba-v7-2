# Active Files Reference - USEF-TSE-Mamba V7

## 🎯 Files You'll Actually Use

### Training Workflow

1. **Generate Dataset**
   ```bash
   ./regenerate_dataset_optimized.sh
   ```
   Uses: `src/data/generate_tse_dataset_v7_parallel.py`

2. **Train Model**
   ```bash
   python src/train_v7.py --config configs/config_v7_optimized.yaml
   ```

3. **Run Inference**
   ```bash
   python src/inference_v7.py --config configs/config_v7_optimized.yaml
   ```

### Core Files Only

```
📁 Models (1 file)
└── src/models/USEF_TSE_Mamba_v7_enhanced.py

📁 Data Processing (3 files)
├── src/data/generate_tse_dataset_v7_parallel.py  # Dataset generation
├── src/data/tse_dataset_v7_cached.py            # Data loading
└── src/data/tse_dataset_v7_augmented.py         # Augmentation

📁 Training (1 file)
└── src/train_v7.py

📁 Configuration (2 files)
├── configs/config_v7_optimized.yaml
└── configs/dataset_config_v7_optimized.yaml

📁 Utilities (2 files)
├── src/utils/losses_v7.py
└── src/utils/ema.py

📁 Scripts (1 file)
└── regenerate_dataset_optimized.sh
```

### Total Active Files: **12 files**

## 📝 Quick Commands

### Dataset Generation
```bash
# Fast parallel generation (recommended)
python src/data/generate_tse_dataset_v7_parallel.py \
    --config configs/dataset_config_v7_optimized.yaml \
    --workers 8
```

### Training
```bash
# Start training
python src/train_v7.py --config configs/config_v7_optimized.yaml

# Resume from checkpoint
python src/train_v7.py --config configs/config_v7_optimized.yaml \
    --resume checkpoints/usef_tse_v7_optimized/checkpoint_epoch_50.pth
```

### Monitoring
```bash
tensorboard --logdir outputs/usef_tse_v7_optimized
```

## ⚠️ Don't Use These (Archived)

- ❌ `USEF_TSE_Mamba_v7.py` - Use enhanced version
- ❌ `generate_tse_dataset_v7.py` - Use parallel version
- ❌ `tse_dataset_v7.py` - Use cached version
- ❌ `config_v7.yaml` - Use optimized config
- ❌ Any files in `_archived_v7_cleanup/`

## 🔧 If You Need To...

### Change batch size
Edit: `configs/config_v7_optimized.yaml` → `data.batch_size`

### Adjust validation frequency
Edit: `configs/config_v7_optimized.yaml` → `training.validation_frequency`

### Modify augmentation
Edit: `configs/config_v7_optimized.yaml` → `training.augmentation`

### Debug memory issues
Reduce: `data.num_workers` or `data.batch_size`

---
*Last updated after cleanup - all legacy files moved to `_archived_v7_cleanup/`*