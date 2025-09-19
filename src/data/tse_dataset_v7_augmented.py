"""
TSE Dataset V7 with Comprehensive Data Augmentation
Implements MixUp, SpecAugment, speed perturbation, and dynamic noise for regularization
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings
import random
import torchaudio.transforms as T

warnings.filterwarnings('ignore')


class TSEDatasetV7Augmented(Dataset):
    """
    Target Speaker Extraction Dataset V7 with augmentation
    Prevents overfitting through comprehensive data augmentation
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        sample_rate: int = 8000,
        segment_length: float = 4.0,
        ref_length_min: float = 2.5,
        ref_length_max: float = 2.5,
        use_metadata: bool = True,
        metadata_path: Optional[str] = None,
        # Augmentation parameters
        use_augmentation: bool = True,
        mixup_alpha: float = 0.2,
        speed_perturb_range: float = 0.1,
        use_spec_augment: bool = True,
        freq_mask_param: int = 15,
        time_mask_param: int = 20,
        noise_augment: bool = True,
        noise_snr_range: Tuple[float, float] = (-5, 15),
        # Standard parameters
        normalize: bool = True,
        normalize_scale: float = 0.95,
        default_speaker_snr: float = 5.0,
        default_noise_snr: float = 10.0,
        return_paths: bool = False,
        **kwargs
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        self.ref_samples = int(ref_length_max * sample_rate)

        # Augmentation settings (only for training)
        self.use_augmentation = use_augmentation and (split == 'train')
        self.mixup_alpha = mixup_alpha
        self.speed_perturb_range = speed_perturb_range
        self.use_spec_augment = use_spec_augment
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.noise_augment = noise_augment
        self.noise_snr_range = noise_snr_range

        # Standard settings
        self.normalize = normalize
        self.normalize_scale = normalize_scale
        self.default_speaker_snr = default_speaker_snr
        self.default_noise_snr = default_noise_snr
        self.return_paths = return_paths

        # Load dataset
        if use_metadata and metadata_path:
            self._load_from_metadata(metadata_path)
        else:
            self._load_from_folders()

        print(f"Loaded {len(self.samples)} samples for {split}")
        if self.use_augmentation:
            print(f"Augmentation enabled: MixUp(α={mixup_alpha}), Speed(±{speed_perturb_range*100}%), "
                  f"SpecAugment({freq_mask_param},{time_mask_param}), Noise({noise_snr_range}dB)")

    def _load_from_metadata(self, metadata_path: str):
        """Load dataset from CSV metadata file"""
        csv_path = Path(metadata_path) / f"{self.split}_metadata.csv"
        if not csv_path.exists():
            csv_path = Path(metadata_path) / f"{self.split}.csv"

        df = pd.read_csv(csv_path)
        self.samples = []

        for _, row in df.iterrows():
            self.samples.append({
                'mixture_path': row['mixture_path'],
                'target_path': row['target_path'],
                'reference_path': row['reference_path'],
                'speaker_snr': row.get('speaker_snr', self.default_speaker_snr),
                'noise_snr': row.get('noise_snr', self.default_noise_snr)
            })

    def _load_from_folders(self):
        """Load dataset from folder structure"""
        split_dir = self.data_root / self.split
        mixture_dir = split_dir / 'mixture'
        target_dir = split_dir / 'target'
        reference_dir = split_dir / 'reference'

        mixture_files = sorted(list(mixture_dir.glob('*.wav')))

        self.samples = []
        for mix_file in mixture_files:
            base_name = mix_file.stem
            target_file = target_dir / f"{base_name}.wav"
            ref_file = reference_dir / f"{base_name}.wav"

            if target_file.exists() and ref_file.exists():
                self.samples.append({
                    'mixture_path': str(mix_file),
                    'target_path': str(target_file),
                    'reference_path': str(ref_file),
                    'speaker_snr': self.default_speaker_snr,
                    'noise_snr': self.default_noise_snr
                })

    def _load_audio_safe(self, path: str, target_samples: int) -> torch.Tensor:
        """Load audio with exact size guarantee"""
        try:
            audio, sr = sf.read(path)

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Ensure correct length
            if len(audio) > target_samples:
                start = np.random.randint(0, max(1, len(audio) - target_samples))
                audio = audio[start:start + target_samples]
            elif len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')

            audio = torch.from_numpy(audio.astype(np.float32))
            return audio.reshape(1, -1)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(1, target_samples, dtype=torch.float32)

    def _apply_speed_perturbation(self, audio: torch.Tensor, factor: float) -> torch.Tensor:
        """Apply speed perturbation using resampling"""
        if factor == 1.0:
            return audio

        # Resample to change speed
        orig_length = audio.shape[-1]
        new_sample_rate = int(self.sample_rate * factor)

        # Use linear interpolation for speed change
        audio_stretched = F.interpolate(
            audio.unsqueeze(0),
            size=int(orig_length / factor),
            mode='linear',
            align_corners=False
        ).squeeze(0)

        # Restore original length
        if audio_stretched.shape[-1] > orig_length:
            audio_stretched = audio_stretched[:, :orig_length]
        elif audio_stretched.shape[-1] < orig_length:
            pad_length = orig_length - audio_stretched.shape[-1]
            audio_stretched = F.pad(audio_stretched, (0, pad_length))

        return audio_stretched

    def _apply_spec_augment(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment-style masking in time domain"""
        if not self.use_spec_augment:
            return audio

        audio = audio.clone()
        length = audio.shape[-1]

        # Time masking - zero out random segments
        if self.time_mask_param > 0:
            num_masks = random.randint(1, 2)
            for _ in range(num_masks):
                mask_length = random.randint(1, min(self.time_mask_param * 80, length // 10))
                mask_start = random.randint(0, max(1, length - mask_length))
                audio[:, mask_start:mask_start + mask_length] *= 0.1  # Reduce instead of zero

        # Frequency masking (simplified for time domain)
        if self.freq_mask_param > 0 and random.random() < 0.5:
            # Apply high-pass or low-pass filtering randomly
            cutoff = random.uniform(0.1, 0.4)
            if random.random() < 0.5:
                # Simple low-pass filter
                audio = audio * cutoff
            else:
                # Simple high-pass effect
                audio = audio * (1 - cutoff) + audio

        return audio

    def _add_noise(self, audio: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add Gaussian noise at specified SNR"""
        signal_power = torch.mean(audio ** 2)
        noise = torch.randn_like(audio)
        noise_power = torch.mean(noise ** 2)

        # Calculate noise scaling factor for target SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_scaling = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-8))

        noisy_audio = audio + noise_scaling * noise
        return noisy_audio

    def _mixup(self, mixture1: torch.Tensor, target1: torch.Tensor, ref1: torch.Tensor,
               mixture2: torch.Tensor, target2: torch.Tensor, ref2: torch.Tensor,
               alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply MixUp augmentation"""
        lambda_mix = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

        # Mix the signals
        mixture = lambda_mix * mixture1 + (1 - lambda_mix) * mixture2
        target = lambda_mix * target1 + (1 - lambda_mix) * target2
        # Keep reference from the dominant speaker
        reference = ref1 if lambda_mix > 0.5 else ref2

        return mixture, target, reference

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get augmented sample"""
        sample = self.samples[idx]

        # Load audio
        mixture = self._load_audio_safe(sample['mixture_path'], self.segment_samples)
        target = self._load_audio_safe(sample['target_path'], self.segment_samples)
        reference = self._load_audio_safe(sample['reference_path'], self.ref_samples)

        # Apply augmentations only during training
        if self.use_augmentation:
            # Speed perturbation
            if self.speed_perturb_range > 0 and random.random() < 0.8:
                speed_factor = 1.0 + random.uniform(-self.speed_perturb_range, self.speed_perturb_range)
                mixture = self._apply_speed_perturbation(mixture, speed_factor)
                target = self._apply_speed_perturbation(target, speed_factor)
                # Don't perturb reference to maintain speaker identity

            # SpecAugment
            if self.use_spec_augment and random.random() < 0.7:
                mixture = self._apply_spec_augment(mixture)
                # Don't augment target to preserve clean signal

            # Dynamic noise addition
            if self.noise_augment and random.random() < 0.5:
                noise_snr = random.uniform(*self.noise_snr_range)
                mixture = self._add_noise(mixture, noise_snr)

            # MixUp (probability 0.3)
            if self.mixup_alpha > 0 and random.random() < 0.3:
                # Get another random sample
                idx2 = random.randint(0, len(self.samples) - 1)
                sample2 = self.samples[idx2]

                mixture2 = self._load_audio_safe(sample2['mixture_path'], self.segment_samples)
                target2 = self._load_audio_safe(sample2['target_path'], self.segment_samples)
                reference2 = self._load_audio_safe(sample2['reference_path'], self.ref_samples)

                mixture, target, reference = self._mixup(
                    mixture, target, reference,
                    mixture2, target2, reference2,
                    self.mixup_alpha
                )

        # Normalize if needed
        if self.normalize:
            max_val = max(
                mixture.abs().max(),
                target.abs().max(),
                reference.abs().max()
            ) + 1e-8
            if max_val > self.normalize_scale:
                scale = self.normalize_scale / max_val
                mixture = mixture * scale
                target = target * scale
                reference = reference * scale

        result = {
            'mixture': mixture.clone(),
            'target': target.clone(),
            'reference': reference.clone(),
            'speaker_snr': torch.tensor(sample.get('speaker_snr', self.default_speaker_snr), dtype=torch.float32),
            'noise_snr': torch.tensor(sample.get('noise_snr', self.default_noise_snr), dtype=torch.float32)
        }

        if self.return_paths:
            result['paths'] = {
                'mixture': sample['mixture_path'],
                'target': sample['target_path'],
                'reference': sample['reference_path']
            }

        return result


def create_augmented_dataloaders(
    data_root: str,
    config: Dict,
    batch_size: int = 2,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create augmented dataloaders for training, validation, and test"""

    data_config = config.get('data', {})
    aug_config = config.get('training', {}).get('augmentation', {})

    # Training with augmentation
    train_dataset = TSEDatasetV7Augmented(
        data_root=data_root,
        split='train',
        sample_rate=data_config.get('sample_rate', 8000),
        segment_length=data_config.get('segment_length', 4.0),
        ref_length_min=data_config.get('ref_length_min', 2.5),
        ref_length_max=data_config.get('ref_length_max', 2.5),
        use_metadata=data_config.get('use_metadata', True),
        metadata_path=data_config.get('metadata_path'),
        # Augmentation parameters
        use_augmentation=config.get('training', {}).get('use_augmentation', True),
        mixup_alpha=aug_config.get('mixup_alpha', 0.2),
        speed_perturb_range=aug_config.get('speed_perturb', 0.1),
        use_spec_augment=aug_config.get('spec_augment', True),
        freq_mask_param=aug_config.get('freq_mask_param', 15),
        time_mask_param=aug_config.get('time_mask_param', 20),
        noise_augment=aug_config.get('noise_augment', True),
        noise_snr_range=tuple(aug_config.get('noise_snr_range', [-5, 15]))
    )

    # Validation without augmentation
    val_dataset = TSEDatasetV7Augmented(
        data_root=data_root,
        split='validation',
        sample_rate=data_config.get('sample_rate', 8000),
        segment_length=data_config.get('segment_length', 4.0),
        ref_length_min=data_config.get('ref_length_min', 2.5),
        ref_length_max=data_config.get('ref_length_max', 2.5),
        use_metadata=data_config.get('use_metadata', True),
        metadata_path=data_config.get('metadata_path'),
        use_augmentation=False  # No augmentation for validation
    )

    # Test without augmentation
    test_dataset = TSEDatasetV7Augmented(
        data_root=data_root,
        split='test',
        sample_rate=data_config.get('sample_rate', 8000),
        segment_length=data_config.get('segment_length', 4.0),
        ref_length_min=data_config.get('ref_length_min', 2.5),
        ref_length_max=data_config.get('ref_length_max', 2.5),
        use_metadata=data_config.get('use_metadata', True),
        metadata_path=data_config.get('metadata_path'),
        use_augmentation=False  # No augmentation for test
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=data_config.get('pin_memory', True),
        drop_last=data_config.get('drop_last_train', True),
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=data_config.get('pin_memory', True),
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=data_config.get('pin_memory', True),
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, test_loader