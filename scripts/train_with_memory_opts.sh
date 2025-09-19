#!/bin/bash
# Training script with memory optimizations

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate usef_tse

# Set PyTorch memory configuration to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Clear any existing cache
python -c "import torch; torch.cuda.empty_cache()"

# Optional: Set to use less memory for CUDA operations
export CUDA_LAUNCH_BLOCKING=1  # Helps debug but slightly slower

# Start training with optimized settings
echo "Starting training with memory optimizations..."
echo "Configuration:"
echo "  - Batch size: 1"
echo "  - Segment length: 3.0 seconds"
echo "  - Reference length: 3.0 seconds (fixed)"
echo "  - Gradient accumulation: 4 (effective batch=4)"
echo "  - Mixed precision: enabled"
echo ""

python src/train_v7.py --config configs/config_v7.yaml

# Alternative: If still OOM, use this command with even more aggressive memory management
# python -u src/train_v7.py --config configs/config_v7.yaml 2>&1 | tee training.log