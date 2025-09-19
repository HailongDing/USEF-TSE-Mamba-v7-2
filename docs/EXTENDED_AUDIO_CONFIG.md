# Extended Audio Configuration for USEF-TSE-Mamba v7

## Configuration Update Summary

### Previous Configuration (v6/v7 initial)
- **Mixture**: 4.0 seconds (32,000 samples)
- **Reference**: 2.5 seconds (20,000 samples)
- **Memory Usage**: 9.8 GB

### New Configuration (v7 extended - Option A)
- **Mixture**: 6.0 seconds (48,000 samples) - 50% increase
- **Reference**: 4.0 seconds (32,000 samples) - 60% increase
- **Memory Usage**: 14.2 GB (verified)
- **Memory Buffer**: 9.5 GB remaining

## Benefits of Extended Audio

### 1. Better Temporal Context
- **6-second mixtures** capture more complete utterances and speaker turns
- Allows model to learn longer-range dependencies
- Better handling of overlapping speech patterns

### 2. Richer Speaker Characteristics
- **4-second references** provide more comprehensive speaker voice samples
- Captures speaking style, not just voice timbre
- More robust to intra-speaker variability

### 3. Improved Generalization
- Longer sequences reduce the risk of memorizing short patterns
- Forces model to learn more abstract speaker representations
- Better performance on real-world variable-length audio

## Memory Analysis

```
Configuration Test Results:
- Peak Memory: 14.19 GB / 23.6 GB (60% usage)
- With Augmentation: 14.29 GB (60.5% usage)
- Safe Buffer: 9.5 GB

Memory Breakdown:
- Model + EMA: 0.2 GB
- Forward Pass: 14.05 GB
- Backward Pass: 14.19 GB (peak)
- Augmentation Overhead: 0.1 GB
```

## Files Updated

1. **`configs/config_v7.yaml`**
   - `segment_length: 4.0` → `6.0`
   - `ref_length_min/max: 2.5` → `4.0`

2. **`configs/dataset_config_v7.yaml`**
   - `duration: 4.0` → `6.0`
   - `ref_duration: 2.5` → `4.0`

## Dataset Regeneration Required

Since audio lengths have changed, you need to regenerate the dataset:

```bash
# Generate new dataset with extended sequences
cd /home/ubuntu/latest/USEF-TSE-Mamba-v7
python src/data/generate_tse_dataset_v7.py --config configs/dataset_config_v7.yaml
```

This will create:
- Training: 20,000 samples of 6s mixtures with 4s references
- Validation: 5,000 samples
- Test: 3,000 samples

**Estimated time**: 3-5 hours (longer due to extended audio)
**Disk space**: ~300 GB (50% more than before)

## Training Adjustments

With longer sequences, consider:

1. **Learning Rate**: May need slight reduction
   - Current: 0.0001
   - If unstable: try 0.00008

2. **Warmup**: More important with longer sequences
   - Already set to 5 epochs

3. **Expected Training Time**:
   - ~40% slower per epoch due to longer sequences
   - But may need fewer epochs to converge

## Expected Performance Impact

### Positive Effects
- **+1-2 dB SI-SDRi** from better temporal modeling
- Better handling of continuous speech
- More robust to real-world scenarios
- Improved speaker discrimination

### Potential Challenges
- Slightly slower training (40% per epoch)
- More disk space for dataset
- May need learning rate tuning

## Next Steps

1. **Regenerate Dataset** (Required)
   ```bash
   python src/data/generate_tse_dataset_v7.py --config configs/dataset_config_v7.yaml
   ```

2. **Start Training**
   ```bash
   python src/train_v7.py --config configs/config_v7.yaml
   ```

3. **Monitor for**
   - Memory usage staying under 15 GB
   - Stable loss curves
   - Improved validation SI-SDRi (target: 11-12 dB)

## Alternative Configurations Tested

| Option | Mixture | Reference | Memory | Status |
|--------|---------|-----------|--------|---------|
| Safe (A) | 6s | 4s | 14.2 GB | ✅ Implemented |
| Aggressive (B) | 8s | 5s | ~20-22 GB | Possible |
| Maximum (C) | 10s | 6s | ~26-28 GB | ❌ Exceeds |

## Conclusion

The extended audio configuration (6s/4s) is:
- ✅ **Safe**: Uses only 60% of GPU memory
- ✅ **Beneficial**: Better temporal and speaker modeling
- ✅ **Practical**: Good balance of performance and resources

This configuration should help achieve the target of **11-12 dB SI-SDRi** with the enhanced model and anti-overfitting strategies.