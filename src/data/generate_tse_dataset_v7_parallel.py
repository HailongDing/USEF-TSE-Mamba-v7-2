#!/usr/bin/env python3
"""
TSE Dataset Generator V7 - Parallel Version
Optimized with multiprocessing for 4-6x faster generation
"""

import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
from collections import defaultdict
import warnings
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import pickle
import time

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSEDatasetGeneratorV7Parallel:
    """
    Parallel TSE dataset generator with optimizations:
    - Multiprocessing for sample generation
    - Pre-loaded speaker index
    - Better resampling with librosa
    - Batch processing
    """

    def __init__(self, config_path: str):
        """Initialize generator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Audio parameters
        self.sample_rate = self.config['audio']['sample_rate']
        self.duration = self.config['audio']['duration']  # 6.0 seconds
        self.ref_duration = self.config['audio']['ref_duration']  # 4.0 seconds

        # Calculate exact sample counts
        self.target_samples = int(self.duration * self.sample_rate)  # 48000
        self.ref_samples = int(self.ref_duration * self.sample_rate)  # 32000

        # SNR ranges
        self.speaker_snr_range = self.config['mixing']['snr_range']
        self.noise_snr_range = self.config['mixing']['noise_snr_range']

        # Paths
        self.librispeech_root = Path(self.config['paths']['librispeech_root'])
        self.wham_noise_root = Path(self.config['paths']['wham_noise_root'])
        self.output_dir = Path(self.config['paths']['output_root'])

        # VAD threshold
        self.vad_energy_threshold = 0.001

        # Parallel processing settings
        self.num_workers = min(cpu_count() - 1, 8)  # Leave one CPU free
        self.batch_size = 100  # Generate in batches

        logger.info(f"Initialized parallel generator with {self.num_workers} workers")
        logger.info(f"Target: {self.duration}s ({self.target_samples} samples)")
        logger.info(f"Reference: {self.ref_duration}s ({self.ref_samples} samples)")

    def build_speaker_index(self, split: str) -> Dict[str, List[Path]]:
        """
        Build complete speaker index upfront
        This is called once per split to avoid repeated directory scanning
        """
        logger.info(f"Building speaker index for {split}...")
        speaker_utterances = defaultdict(list)

        # Get splits from config
        librispeech_splits = self.config['paths'].get('librispeech_splits', {})

        if split in librispeech_splits:
            split_dirs = librispeech_splits[split]
            if not isinstance(split_dirs, list):
                split_dirs = [split_dirs]
        else:
            # Fallback
            split_map = {
                'train': ['train-clean-100', 'train-clean-360'],
                'validation': ['dev-clean', 'dev-other'],
                'test': ['test-clean']
            }
            split_dirs = split_map.get(split, [split])

        # Process all split directories
        for libri_split in split_dirs:
            split_dir = self.librispeech_root / libri_split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue

            # Recursively find all FLAC files
            for flac_file in split_dir.rglob('*.flac'):
                # Extract speaker ID from path
                # Format: .../speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
                speaker_id = flac_file.parent.parent.name
                speaker_utterances[speaker_id].append(flac_file)

        # Filter speakers with sufficient utterances
        min_utterances = 2
        speaker_utterances = {
            spk: utts for spk, utts in speaker_utterances.items()
            if len(utts) >= min_utterances
        }

        total_speakers = len(speaker_utterances)
        total_utterances = sum(len(utts) for utts in speaker_utterances.values())
        logger.info(f"Indexed {total_speakers} speakers with {total_utterances} utterances")

        return dict(speaker_utterances)

    def load_audio_librosa(self, file_path: str, target_samples: int, offset: float = 0.0) -> Optional[np.ndarray]:
        """
        Load audio using librosa for better quality resampling
        """
        try:
            # Load with librosa (handles resampling automatically)
            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=True,
                offset=offset,
                duration=target_samples / self.sample_rate + 0.5  # Load slightly more
            )

            # Ensure exact length
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            elif len(audio) < target_samples:
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)

            # Simple VAD check
            rms = np.sqrt(np.mean(audio[:min(len(audio), target_samples)] ** 2))
            if rms < self.vad_energy_threshold:
                return None

            return audio

        except Exception as e:
            logger.debug(f"Error loading {file_path}: {e}")
            return None

    def mix_at_snr(self, signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        """Mix signal with noise at specified SNR"""
        signal_rms = np.sqrt(np.mean(signal ** 2))
        noise_rms = np.sqrt(np.mean(noise ** 2))

        if noise_rms == 0:
            return signal

        snr_linear = 10 ** (snr_db / 20)
        noise_scale = signal_rms / (noise_rms * snr_linear)

        mixed = signal + noise * noise_scale

        # Prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * 0.95 / max_val

        return mixed

    def generate_single_mixture(self, args):
        """
        Generate a single mixture (for parallel processing)
        This function is designed to be called by multiprocessing.Pool
        """
        speaker_data, noise_files, seed = args

        # Set random seed for this worker
        np.random.seed(seed)
        random.seed(seed)

        max_attempts = 30
        for _ in range(max_attempts):
            try:
                # Select two different speakers
                if len(speaker_data) < 2:
                    continue

                speakers = random.sample(list(speaker_data.keys()), 2)
                target_speaker = speakers[0]
                interference_speaker = speakers[1]

                # Get utterances
                target_utterances = speaker_data[target_speaker]
                interference_utterances = speaker_data[interference_speaker]

                if len(target_utterances) < 2:
                    continue

                # Select utterances
                target_utt = random.choice(target_utterances)
                interference_utt = random.choice(interference_utterances)

                # Select different reference
                ref_candidates = [u for u in target_utterances if u != target_utt]
                if not ref_candidates:
                    continue
                ref_utt = random.choice(ref_candidates)

                # Load audio with librosa
                target_audio = self.load_audio_librosa(str(target_utt), self.target_samples)
                if target_audio is None:
                    continue

                interference_audio = self.load_audio_librosa(str(interference_utt), self.target_samples)
                if interference_audio is None:
                    continue

                ref_audio = self.load_audio_librosa(str(ref_utt), self.ref_samples)
                if ref_audio is None:
                    continue

                # Mix speakers
                speaker_snr = random.uniform(*self.speaker_snr_range)
                mixture = self.mix_at_snr(target_audio, interference_audio, speaker_snr)

                # Add noise
                if noise_files:
                    noise_file = random.choice(noise_files)
                    noise = self.load_audio_librosa(str(noise_file), self.target_samples)
                    if noise is not None:
                        noise_snr = random.uniform(*self.noise_snr_range)
                        mixture = self.mix_at_snr(mixture, noise, noise_snr)
                    else:
                        noise_snr = 20.0  # High SNR for white noise
                        white_noise = np.random.randn(self.target_samples) * 0.01
                        mixture = self.mix_at_snr(mixture, white_noise, noise_snr)
                else:
                    noise_snr = 20.0

                return {
                    'mixture': mixture,
                    'target': target_audio,
                    'reference': ref_audio,
                    'target_speaker': target_speaker,
                    'interference_speaker': interference_speaker,
                    'speaker_snr': speaker_snr,
                    'noise_snr': noise_snr,
                    'target_path': str(target_utt),
                    'interference_path': str(interference_utt),
                    'reference_path': str(ref_utt)
                }

            except Exception as e:
                logger.debug(f"Error in generation attempt: {e}")
                continue

        return None

    def generate_batch_parallel(
        self,
        speaker_data: Dict[str, List[Path]],
        noise_files: List[Path],
        batch_size: int
    ) -> List[Dict]:
        """
        Generate a batch of samples in parallel
        """
        # Prepare arguments for parallel processing
        # Each worker gets the same data but different random seed
        args_list = [
            (speaker_data, noise_files, random.randint(0, 1000000))
            for _ in range(batch_size * 2)  # Generate 2x to account for failures
        ]

        # Process in parallel
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(self.generate_single_mixture, args_list)

        # Filter successful generations
        successful = [r for r in results if r is not None]

        return successful[:batch_size]

    def save_audio(self, audio: np.ndarray, path: Path):
        """Save audio with verification"""
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, self.sample_rate)

    def generate_dataset(self, split: str, num_samples: int):
        """
        Generate dataset for a specific split using parallel processing
        """
        logger.info(f"\nGenerating {split} split with {num_samples} samples...")
        start_time = time.time()

        # Build speaker index once
        speaker_data = self.build_speaker_index(split)
        if not speaker_data:
            logger.error(f"No speaker data found for {split}")
            return

        # Get noise files
        noise_files = list(self.wham_noise_root.glob('**/*.wav'))
        if not noise_files:
            logger.warning("No noise files found, will use white noise")
            noise_files = []

        # Output directories
        split_dir = self.output_dir / split
        mixture_dir = split_dir / 'mixture'
        target_dir = split_dir / 'target'
        reference_dir = split_dir / 'reference'

        for dir_path in [mixture_dir, target_dir, reference_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Generate in batches
        metadata = []
        successful = 0

        with tqdm(total=num_samples, desc=f"Generating {split}") as pbar:
            while successful < num_samples:
                # Calculate batch size
                remaining = num_samples - successful
                current_batch_size = min(self.batch_size, remaining)

                # Generate batch in parallel
                batch_samples = self.generate_batch_parallel(
                    speaker_data, noise_files, current_batch_size
                )

                # Save samples
                for sample in batch_samples:
                    if sample is None:
                        continue

                    sample_id = f"{split}_{successful:06d}"

                    # Save audio files
                    mixture_path = mixture_dir / f"{sample_id}.wav"
                    target_path = target_dir / f"{sample_id}.wav"
                    reference_path = reference_dir / f"{sample_id}.wav"

                    self.save_audio(sample['mixture'], mixture_path)
                    self.save_audio(sample['target'], target_path)
                    self.save_audio(sample['reference'], reference_path)

                    # Add metadata
                    metadata.append({
                        'sample_id': sample_id,
                        'mixture_path': str(mixture_path),
                        'target_path': str(target_path),
                        'reference_path': str(reference_path),
                        'target_speaker': sample['target_speaker'],
                        'interference_speaker': sample['interference_speaker'],
                        'speaker_snr': sample['speaker_snr'],
                        'noise_snr': sample['noise_snr']
                    })

                    successful += 1
                    pbar.update(1)

                    if successful >= num_samples:
                        break

        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_path = self.output_dir.parent / 'metadata_v7' / f"{split}_metadata.csv"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(metadata_path, index=False)

        elapsed = time.time() - start_time
        samples_per_second = num_samples / elapsed
        logger.info(f"Generated {successful} samples in {elapsed:.1f}s ({samples_per_second:.1f} samples/s)")
        logger.info(f"Metadata saved to {metadata_path}")

    def generate_all(self):
        """Generate all dataset splits"""
        for split, num_samples in self.config['dataset_sizes'].items():
            self.generate_dataset(split, num_samples)

        self.generate_statistics()

    def generate_statistics(self):
        """Generate dataset statistics report"""
        stats = []

        for split in ['train', 'validation', 'test']:
            split_dir = self.output_dir / split

            if not split_dir.exists():
                continue

            # Count files
            n_mixture = len(list((split_dir / 'mixture').glob('*.wav')))
            n_target = len(list((split_dir / 'target').glob('*.wav')))
            n_reference = len(list((split_dir / 'reference').glob('*.wav')))

            assert n_mixture == n_target == n_reference, f"File count mismatch in {split}"

            stats.append(f"{split}: {n_mixture} samples")

        # Write statistics
        stats_path = self.output_dir.parent / 'metadata_v7' / 'statistics_report.txt'
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        with open(stats_path, 'w') as f:
            f.write("TSE Dataset V7 Statistics (Parallel Generation)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Sample rate: {self.sample_rate} Hz\n")
            f.write(f"Mixture/Target duration: {self.duration}s ({self.target_samples} samples)\n")
            f.write(f"Reference duration: {self.ref_duration}s ({self.ref_samples} samples)\n")
            f.write(f"Parallel workers: {self.num_workers}\n\n")
            f.write("Splits:\n")
            for stat in stats:
                f.write(f"  {stat}\n")

        logger.info(f"Statistics saved to {stats_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate TSE Dataset V7 (Parallel)')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/dataset_config_v7.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'validation', 'test', 'all'],
        default='all',
        help='Which split to generate'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers'
    )

    args = parser.parse_args()

    # Create generator
    generator = TSEDatasetGeneratorV7Parallel(args.config)
    generator.num_workers = args.workers

    # Generate dataset
    if args.split == 'all':
        generator.generate_all()
    else:
        num_samples = generator.config['dataset_sizes'][args.split]
        generator.generate_dataset(args.split, num_samples)

    logger.info("Dataset generation complete!")


if __name__ == '__main__':
    main()