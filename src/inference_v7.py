import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
import soundfile as sf
import logging
import yaml
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple
import pandas as pd

# Using Enhanced model for v7
from models.USEF_TSE_Mamba_v7_enhanced import USEF_TSE_Mamba_v7_Enhanced as USEF_TSE_Mamba_v7
# Use cached dataset for efficient loading
from data.tse_dataset_v7_cached import TSEDatasetV7Cached as TSEDatasetV7
from utils.losses_v7 import si_sdr_loss, calculate_si_sdri

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TSEInferenceV7:
    """
    Enhanced inference engine for USEF-TSE v7
    Supports both single file and batch processing with dataset integration
    """
    
    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None, device=device):
        self.device = device
        
        # Load checkpoint
        log.info(f"Loading v7 model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', {})
        
        # Initialize model
        model_config = self.config.get('model', {})
        self.model = USEF_TSE_Mamba_v7(
            dim=model_config.get('dim', 128),
            kernel_sizes=model_config.get('kernel_sizes', [16, 16, 16]),
            strides=model_config.get('strides', [2, 2, 2]),
            num_blocks=model_config.get('num_blocks', 6),
            act_fn=model_config.get('activation', 'silu'),  # Default to SiLU for enhanced
            num_heads=model_config.get('num_heads', 8),
            dropout=model_config.get('dropout', 0.1),
            use_bottleneck_attention_only=model_config.get('use_bottleneck_attention_only', True)
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        log.info(f"V7 Model loaded. Best SI-SDRi: {checkpoint.get('best_si_sdri', 'N/A'):.2f} dB")
        
        # Get sample rate from config
        self.sample_rate = self.config.get('data', {}).get('sample_rate', 8000)
        
        # Initialize metrics storage
        self.metrics = []
    
    def extract_speaker(
        self, 
        mixture_path: str, 
        reference_path: str, 
        output_path: Optional[str] = None,
        compute_metrics: bool = False,
        target_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int, Optional[Dict]]:
        """
        Extract target speaker from mixture using reference audio
        
        Args:
            mixture_path: Path to mixed audio file
            reference_path: Path to reference audio from target speaker
            output_path: Optional path to save extracted audio
            compute_metrics: Whether to compute SI-SDR metrics
            target_path: Path to clean target (for metric computation)
        
        Returns:
            extracted_audio: Extracted audio as numpy array
            sample_rate: Sample rate
            metrics: Optional dictionary of metrics
        """
        # Load audio files
        mixture, sr_mix = torchaudio.load(mixture_path)
        reference, sr_ref = torchaudio.load(reference_path)
        
        # Resample if needed
        if sr_mix != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr_mix, self.sample_rate)
            mixture = resampler(mixture)
        
        if sr_ref != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr_ref, self.sample_rate)
            reference = resampler(reference)
        
        # Convert to mono and add batch dimension
        if mixture.shape[0] > 1:
            mixture = mixture.mean(dim=0, keepdim=True)
        if reference.shape[0] > 1:
            reference = reference.mean(dim=0, keepdim=True)
        
        mixture = mixture.unsqueeze(0).to(self.device)
        reference = reference.unsqueeze(0).to(self.device)
        
        # Extract target speaker
        with torch.no_grad():
            extracted = self.model(mixture, reference)
        
        # Compute metrics if requested
        metrics = None
        if compute_metrics and target_path:
            target, sr_tgt = torchaudio.load(target_path)
            if sr_tgt != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr_tgt, self.sample_rate)
                target = resampler(target)
            
            if target.shape[0] > 1:
                target = target.mean(dim=0, keepdim=True)
            target = target.unsqueeze(0).to(self.device)
            
            # Ensure same length
            min_len = min(extracted.shape[-1], target.shape[-1], mixture.shape[-1])
            extracted_trim = extracted[..., :min_len]
            target_trim = target[..., :min_len]
            mixture_trim = mixture[..., :min_len]
            
            # Calculate metrics
            si_sdr = -si_sdr_loss(extracted_trim, target_trim).item()
            si_sdri = calculate_si_sdri(extracted_trim, mixture_trim, target_trim)
            
            metrics = {
                'si_sdr': si_sdr,
                'si_sdri': si_sdri,
                'mixture_path': mixture_path,
                'reference_path': reference_path
            }
            
            self.metrics.append(metrics)
        
        # Convert to numpy
        extracted_np = extracted.squeeze().cpu().numpy()
        
        # Normalize
        max_val = np.max(np.abs(extracted_np))
        if max_val > 0:
            extracted_np = extracted_np / max_val * 0.95
        
        # Save if output path provided
        if output_path:
            sf.write(output_path, extracted_np, self.sample_rate)
            log.info(f"Extracted audio saved to {output_path}")
        
        return extracted_np, self.sample_rate, metrics
    
    def process_dataset(
        self,
        dataset_root: str,
        split: str = 'test',
        output_dir: Optional[str] = None,
        num_samples: Optional[int] = None,
        compute_metrics: bool = True
    ) -> pd.DataFrame:
        """
        Process entire dataset split
        
        Args:
            dataset_root: Root directory of TSE dataset
            split: Dataset split to process ('train', 'validation', 'test')
            output_dir: Directory to save extracted audio
            num_samples: Number of samples to process (None for all)
            compute_metrics: Whether to compute metrics
        
        Returns:
            DataFrame with results and metrics
        """
        # Create dataset
        dataset = TSEDatasetV7(
            data_root=dataset_root,
            split=split,
            sample_rate=self.sample_rate,
            augmentation=False,
            return_paths=True
        )
        
        # Limit samples if specified
        if num_samples:
            indices = list(range(min(num_samples, len(dataset))))
        else:
            indices = list(range(len(dataset)))
        
        # Create output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Process samples
        for idx in tqdm(indices, desc=f"Processing {split} set"):
            sample = dataset[idx]
            
            # Get paths
            paths = sample['paths']
            mixture_path = paths['mixture']
            reference_path = paths['reference']
            target_path = paths['target'] if compute_metrics else None
            
            # Generate output path
            if output_dir:
                sample_id = Path(mixture_path).stem
                output_path = output_dir / f"{sample_id}_extracted.wav"
            else:
                output_path = None
            
            # Extract speaker
            _, _, metrics = self.extract_speaker(
                mixture_path=mixture_path,
                reference_path=reference_path,
                output_path=output_path,
                compute_metrics=compute_metrics,
                target_path=target_path
            )
            
            if metrics:
                results.append(metrics)
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
        if compute_metrics and len(df) > 0:
            log.info(f"\n{split.upper()} Set Results:")
            log.info(f"  Average SI-SDR: {df['si_sdr'].mean():.2f} dB")
            log.info(f"  Average SI-SDRi: {df['si_sdri'].mean():.2f} dB")
            log.info(f"  Std SI-SDRi: {df['si_sdri'].std():.2f} dB")
        
        return df
    
    def batch_inference(
        self,
        csv_path: str,
        output_dir: Optional[str] = None,
        compute_metrics: bool = True
    ) -> pd.DataFrame:
        """
        Process samples from CSV file
        
        CSV should have columns: mixture_path, reference_path, [target_path]
        
        Args:
            csv_path: Path to CSV file with sample paths
            output_dir: Directory to save extracted audio
            compute_metrics: Whether to compute metrics
        
        Returns:
            DataFrame with results
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            mixture_path = row['mixture_path']
            reference_path = row['reference_path']
            target_path = row.get('target_path') if compute_metrics else None
            
            # Generate output path
            if output_dir:
                sample_id = f"sample_{idx:06d}"
                output_path = output_dir / f"{sample_id}_extracted.wav"
            else:
                output_path = None
            
            # Extract speaker
            _, _, metrics = self.extract_speaker(
                mixture_path=mixture_path,
                reference_path=reference_path,
                output_path=output_path,
                compute_metrics=compute_metrics and target_path is not None,
                target_path=target_path
            )
            
            if metrics:
                results.append(metrics)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if compute_metrics and len(results_df) > 0:
            log.info(f"\nBatch Results:")
            log.info(f"  Processed: {len(results_df)} samples")
            log.info(f"  Average SI-SDR: {results_df['si_sdr'].mean():.2f} dB")
            log.info(f"  Average SI-SDRi: {results_df['si_sdri'].mean():.2f} dB")
        
        return results_df
    
    def save_metrics(self, output_path: str):
        """Save accumulated metrics to CSV"""
        if self.metrics:
            df = pd.DataFrame(self.metrics)
            df.to_csv(output_path, index=False)
            log.info(f"Metrics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="USEF-TSE-Mamba V7 Inference")
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to v7 model checkpoint')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    
    # Single file mode
    parser.add_argument('--mixture', type=str,
                        help='Path to mixed audio file')
    parser.add_argument('--reference', type=str,
                        help='Path to reference audio')
    parser.add_argument('--target', type=str,
                        help='Path to target audio (for metrics)')
    parser.add_argument('--output', type=str,
                        help='Path to save extracted audio')
    
    # Dataset mode
    parser.add_argument('--dataset', type=str,
                        help='Path to TSE dataset root')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to process')
    
    # Batch mode
    parser.add_argument('--csv', type=str,
                        help='Path to CSV file for batch processing')
    
    # Output options
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save extracted audio files')
    parser.add_argument('--metrics_file', type=str,
                        help='Path to save metrics CSV')
    
    # Processing options
    parser.add_argument('--num_samples', type=int,
                        help='Number of samples to process')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute SI-SDR metrics')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = TSEInferenceV7(args.checkpoint, args.config)
    
    # Process based on mode
    if args.dataset:
        # Dataset mode
        results = inference.process_dataset(
            dataset_root=args.dataset,
            split=args.split,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            compute_metrics=args.compute_metrics
        )
        
        if args.metrics_file:
            results.to_csv(args.metrics_file, index=False)
            log.info(f"Results saved to {args.metrics_file}")
    
    elif args.csv:
        # Batch mode from CSV
        results = inference.batch_inference(
            csv_path=args.csv,
            output_dir=args.output_dir,
            compute_metrics=args.compute_metrics
        )
        
        if args.metrics_file:
            results.to_csv(args.metrics_file, index=False)
    
    elif args.mixture and args.reference:
        # Single file mode
        extracted, sr, metrics = inference.extract_speaker(
            mixture_path=args.mixture,
            reference_path=args.reference,
            output_path=args.output,
            compute_metrics=args.compute_metrics,
            target_path=args.target
        )
        
        log.info(f"Extraction completed")
        log.info(f"  Sample rate: {sr} Hz")
        log.info(f"  Duration: {len(extracted)/sr:.2f} seconds")
        
        if metrics:
            log.info(f"  SI-SDR: {metrics['si_sdr']:.2f} dB")
            log.info(f"  SI-SDRi: {metrics['si_sdri']:.2f} dB")
        
        if args.metrics_file:
            inference.save_metrics(args.metrics_file)
    
    else:
        parser.error("Please specify either --dataset, --csv, or --mixture and --reference")


if __name__ == "__main__":
    main()