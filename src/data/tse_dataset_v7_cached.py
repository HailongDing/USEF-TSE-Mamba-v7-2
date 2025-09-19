"""
TSE Dataset V7 with LRU Caching
Optimized version with audio file caching for faster loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')


class TSEDatasetV7Cached(Dataset):
    """
    Target Speaker Extraction Dataset V7 with LRU caching
    Reduces I/O overhead by caching frequently accessed audio files
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        sample_rate: int = 8000,
        segment_length: float = 6.0,
        ref_length_min: float = 4.0,
        ref_length_max: float = 4.0,
        use_metadata: bool = True,
        metadata_path: Optional[str] = None,
        normalize: bool = False,
        normalize_scale: float = 0.95,
        default_speaker_snr: float = 5.0,
        default_noise_snr: float = 10.0,
        return_paths: bool = False,
        cache_size: int = 1024,  # Number of files to cache
        **kwargs
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.ref_samples = int(ref_length_max * sample_rate)
        self.normalize = normalize
        self.normalize_scale = normalize_scale
        self.default_speaker_snr = default_speaker_snr
        self.default_noise_snr = default_noise_snr
        self.return_paths = return_paths
        self.cache_size = cache_size

        # Setup LRU cache for audio loading
        self._setup_cache()

        # Load dataset
        if use_metadata and metadata_path:
            self._load_from_metadata(metadata_path)
        else:
            self._load_from_folders()

        print(f"Loaded {len(self.samples)} samples for {split} with LRU cache size {cache_size}")

    def _setup_cache(self):
        """Setup LRU cache for audio loading"""
        # Create cached version of load function
        self._load_audio_cached = lru_cache(maxsize=self.cache_size)(self._load_audio_raw)

    def _load_from_metadata(self, metadata_path: str):
        """Load dataset from CSV metadata file"""
        csv_path = Path(metadata_path) / f"{self.split}_metadata.csv"
        if not csv_path.exists():
            csv_path = Path(metadata_path) / f"{self.split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}/{self.split}_metadata.csv")

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

        if not mixture_dir.exists():
            raise FileNotFoundError(f"Mixture directory not found: {mixture_dir}")

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

    def _load_audio_raw(self, path: str) -> Tuple[np.ndarray, int]:
        """
        Raw audio loading function (will be cached)
        Returns tuple for hashability in cache
        """
        try:
            audio, sr = sf.read(path)

            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            return audio, sr
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(1), self.sample_rate

    def _process_audio(self, path: str, target_samples: int) -> torch.Tensor:
        """
        Process cached audio to correct size and format
        """
        # Get from cache
        audio, sr = self._load_audio_cached(path)

        # Make a copy to avoid modifying cached data
        audio = audio.copy()

        # Ensure correct length
        if len(audio) > target_samples:
            # Random crop for training, center crop for val/test
            if self.split == 'train':
                start = np.random.randint(0, max(1, len(audio) - target_samples))
                audio = audio[start:start + target_samples]
            else:
                start = (len(audio) - target_samples) // 2
                audio = audio[start:start + target_samples]
        elif len(audio) < target_samples:
            # Pad with zeros
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)

        # Convert to tensor
        audio = audio.astype(np.float32)
        tensor = torch.from_numpy(audio).reshape(1, -1)
        tensor = tensor.contiguous()

        # Verify size
        assert tensor.shape == (1, target_samples), f"Shape mismatch: {tensor.shape}"

        return tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with caching"""
        sample = self.samples[idx]

        # Load with caching
        mixture = self._process_audio(sample['mixture_path'], self.segment_samples)
        target = self._process_audio(sample['target_path'], self.segment_samples)
        reference = self._process_audio(sample['reference_path'], self.ref_samples)

        result = {
            'mixture': mixture,
            'target': target,
            'reference': reference,
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

    def clear_cache(self):
        """Clear the LRU cache"""
        self._load_audio_cached.cache_clear()

    def cache_info(self):
        """Get cache statistics"""
        return self._load_audio_cached.cache_info()


def create_tse_dataloaders_cached(
    data_root: str,
    batch_size: int = 2,
    num_workers: int = 16,
    sample_rate: int = 8000,
    segment_length: float = 6.0,
    use_metadata: bool = True,
    metadata_path: Optional[str] = None,
    pin_memory: bool = True,
    ref_length_min: float = 4.0,
    ref_length_max: float = 4.0,
    normalize: bool = False,
    normalize_scale: float = 0.95,
    default_speaker_snr: float = 5.0,
    default_noise_snr: float = 10.0,
    drop_last_train: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    cache_size: int = 1024
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create cached dataloaders"""

    # Create datasets with caching
    train_dataset = TSEDatasetV7Cached(
        data_root=data_root,
        split='train',
        sample_rate=sample_rate,
        segment_length=segment_length,
        ref_length_min=ref_length_min,
        ref_length_max=ref_length_max,
        use_metadata=use_metadata,
        metadata_path=metadata_path,
        normalize=normalize,
        normalize_scale=normalize_scale,
        default_speaker_snr=default_speaker_snr,
        default_noise_snr=default_noise_snr,
        cache_size=cache_size
    )

    val_dataset = TSEDatasetV7Cached(
        data_root=data_root,
        split='validation',
        sample_rate=sample_rate,
        segment_length=segment_length,
        ref_length_min=ref_length_min,
        ref_length_max=ref_length_max,
        use_metadata=use_metadata,
        metadata_path=metadata_path,
        normalize=normalize,
        normalize_scale=normalize_scale,
        default_speaker_snr=default_speaker_snr,
        default_noise_snr=default_noise_snr,
        cache_size=cache_size // 2  # Smaller cache for validation
    )

    test_dataset = TSEDatasetV7Cached(
        data_root=data_root,
        split='test',
        sample_rate=sample_rate,
        segment_length=segment_length,
        ref_length_min=ref_length_min,
        ref_length_max=ref_length_max,
        use_metadata=use_metadata,
        metadata_path=metadata_path,
        normalize=normalize,
        normalize_scale=normalize_scale,
        default_speaker_snr=default_speaker_snr,
        default_noise_snr=default_noise_snr,
        cache_size=cache_size // 4  # Even smaller for test
    )

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(4, num_workers // 2),  # Fewer workers for validation
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(4, num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    return train_loader, val_loader, test_loader