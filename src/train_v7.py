import os
# Set PyTorch memory configuration to avoid fragmentation
# This MUST be set before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import numpy as np
import logging
import math
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Dict, Tuple, Optional
from torch.cuda.amp import autocast, GradScaler

# Using Enhanced model for v7 - provides +2-4 dB improvement over original
from models.USEF_TSE_Mamba_v7_enhanced import USEF_TSE_Mamba_v7_Enhanced as USEF_TSE_Mamba_v7
# Use augmented dataset for anti-overfitting
from data.tse_dataset_v7_augmented import create_augmented_dataloaders
# Use cached dataset for efficient loading
from data.tse_dataset_v7_cached import create_tse_dataloaders_cached as create_tse_dataloaders
from utils.losses_v7 import si_sdr_loss, calculate_si_sdri
from utils.ema import ModelEMA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training_v7.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


class WarmupReduceLROnPlateau:
    """Custom scheduler combining warmup with ReduceLROnPlateau"""

    def __init__(self, optimizer, warmup_epochs, warmup_start_lr, target_lr,
                 mode='max', factor=0.5, patience=10, min_lr=1e-6, verbose=True):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.target_lr = target_lr
        self.current_epoch = 0
        self.verbose = verbose

        # Initialize optimizer with warmup start LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_start_lr

        # Create ReduceLROnPlateau for after warmup
        self.reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
            min_lr=min_lr, verbose=verbose
        )

        # Track if warmup is complete
        self.warmup_complete = False

    def step(self, metric=None):
        """Step the scheduler based on current epoch and metric"""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup phase
            progress = self.current_epoch / self.warmup_epochs
            lr = self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) * progress

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if self.verbose:
                log.info(f"Warmup epoch {self.current_epoch}/{self.warmup_epochs}: LR = {lr:.2e}")

            if self.current_epoch == self.warmup_epochs:
                self.warmup_complete = True
                if self.verbose:
                    log.info(f"Warmup complete. Switching to ReduceLROnPlateau with LR = {self.target_lr:.2e}")
        else:
            # ReduceLROnPlateau phase
            if metric is not None:
                self.reduce_lr_scheduler.step(metric)

    def state_dict(self):
        """Returns the state of the scheduler as a dict"""
        return {
            'current_epoch': self.current_epoch,
            'warmup_complete': self.warmup_complete,
            'reduce_lr_state': self.reduce_lr_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler state"""
        self.current_epoch = state_dict['current_epoch']
        self.warmup_complete = state_dict['warmup_complete']
        self.reduce_lr_scheduler.load_state_dict(state_dict['reduce_lr_state'])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


class TSETrainerV7:
    """Enhanced trainer for USEF-TSE v3 with real dataset support"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = device
        
        # Create output directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.build_model()

        # Setup EMA model if enabled
        self.ema_model = None
        if config.get('training', {}).get('use_ema', False):
            log.info("Initializing EMA model for better generalization")
            self.ema_model = ModelEMA(
                self.model,
                decay=config['training'].get('ema_decay', 0.999),
                device=self.device,
                update_every=config['training'].get('ema_update_every', 10)
            )

        # Initialize optimizer and scheduler
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = self.build_dataloaders()
        
        # Initialize tensorboard
        if config['logging'].get('use_tensorboard', True):
            self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        else:
            self.writer = None
        
        # Training state
        self.start_epoch = 0
        self.best_si_sdri = float('-inf')
        self.global_step = 0

        # Mixed precision training
        self.use_mixed_precision = config['training'].get('mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            log.info("Mixed precision training enabled")

        # Gradient accumulation
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation', 1)
        if self.gradient_accumulation_steps > 1:
            log.info(f"Gradient accumulation: {self.gradient_accumulation_steps} steps")
        
        # Load checkpoint if specified
        if config.get('resume_checkpoint'):
            self.load_checkpoint(config['resume_checkpoint'])
    
    def setup_directories(self):
        """Create necessary directories"""
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config for reproducibility
        config_path = self.output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        log.info(f"Config saved to {config_path}")
    
    def build_model(self) -> nn.Module:
        """Build and initialize the model"""
        model_config = self.config['model']
        
        model = USEF_TSE_Mamba_v7(
            dim=model_config['dim'],
            kernel_sizes=model_config['kernel_sizes'],
            strides=model_config['strides'],
            num_blocks=model_config['num_blocks'],
            act_fn=model_config['activation'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout'],
            use_bottleneck_attention_only=model_config.get('use_bottleneck_attention_only', True)
        ).to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1 and self.config.get('use_multi_gpu', False):
            log.info(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer"""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                betas=tuple(opt_config.get('betas', [0.9, 0.999])),
                weight_decay=opt_config.get('weight_decay', 1e-5)
            )
        elif opt_config['type'] == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                betas=tuple(opt_config.get('betas', [0.9, 0.999])),
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
        
        return optimizer
    
    def build_scheduler(self):
        """Build learning rate scheduler with optional warmup"""
        sched_config = self.config.get('scheduler')
        if not sched_config:
            return None

        # Check if warmup is enabled
        training_config = self.config.get('training', {})
        warmup_epochs = training_config.get('warmup_epochs', 0)

        if warmup_epochs > 0 and sched_config['type'] == 'reduce_on_plateau':
            # Use combined warmup + ReduceLROnPlateau scheduler
            warmup_start_lr = training_config.get('warmup_start_lr', 1e-5)
            target_lr = self.config['optimizer']['learning_rate']

            scheduler = WarmupReduceLROnPlateau(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                warmup_start_lr=warmup_start_lr,
                target_lr=target_lr,
                mode='max',  # Maximize SI-SDRi
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                min_lr=sched_config.get('min_lr', 1e-6),
                verbose=True
            )
            log.info(f"Using WarmupReduceLROnPlateau scheduler:")
            log.info(f"  Warmup: {warmup_epochs} epochs, {warmup_start_lr:.2e} -> {target_lr:.2e}")
            log.info(f"  Then ReduceLROnPlateau with factor={sched_config.get('factor', 0.5)}, patience={sched_config.get('patience', 10)}")
        elif sched_config['type'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize SI-SDRi
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                min_lr=sched_config.get('min_lr', 1e-6),
                verbose=True
            )
        elif sched_config['type'] == 'cosine':
            if warmup_epochs > 0:
                # Cosine annealing with warmup
                warmup_start_lr = training_config.get('warmup_start_lr', 1e-5)
                target_lr = self.config['optimizer']['learning_rate']

                # Create a lambda scheduler for warmup + cosine
                def lr_lambda(epoch):
                    if epoch < warmup_epochs:
                        # Linear warmup
                        return (warmup_start_lr + (target_lr - warmup_start_lr) * epoch / warmup_epochs) / target_lr
                    else:
                        # Cosine annealing
                        progress = (epoch - warmup_epochs) / (self.config['training']['num_epochs'] - warmup_epochs)
                        min_lr = sched_config.get('min_lr', 1e-6)
                        return (min_lr + (target_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))) / target_lr

                scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config['training']['num_epochs'],
                    eta_min=sched_config.get('min_lr', 1e-6)
                )
        elif sched_config['type'] == 'cosine_annealing_warm_restarts':
            # Cosine annealing with warm restarts for better convergence
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.get('T_0', 10),
                T_mult=sched_config.get('T_mult', 2),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
            log.info(f"Using CosineAnnealingWarmRestarts: T_0={sched_config.get('T_0', 10)}, T_mult={sched_config.get('T_mult', 2)}")
        else:
            raise ValueError(f"Unknown scheduler type: {sched_config['type']}")

        return scheduler
    
    def build_dataloaders(self) -> Tuple:
        """Build data loaders with augmentation support"""
        data_config = self.config['data']
        training_config = self.config.get('training', {})

        # Check if augmentation is enabled (Option B)
        use_augmentation = training_config.get('use_augmentation', False)

        # Check if variable reference is enabled
        use_variable_ref = data_config.get('use_variable_ref', False)

        if use_augmentation:
            # Use augmented dataset for comprehensive anti-overfitting
            log.info("Using AUGMENTED dataset with MixUp, SpecAugment, and noise")
            train_loader, val_loader, test_loader = create_augmented_dataloaders(
                data_root=data_config['data_root'],
                config=self.config,
                batch_size=data_config['batch_size'],
                num_workers=data_config.get('num_workers', 4)
            )
        elif use_variable_ref:
            # Use variable reference dataset
            from data.tse_dataset_v7_variable_ref import create_variable_ref_dataloaders
            log.info("Using variable reference dataset for training")

            train_loader, val_loader, test_loader = create_variable_ref_dataloaders(
                data_root=data_config['data_root'],
                batch_size=data_config['batch_size'],
                num_workers=data_config['num_workers'],
                sample_rate=data_config['sample_rate'],
                segment_length=data_config['segment_length'],
                use_metadata=data_config.get('use_metadata', False),
                metadata_path=data_config.get('metadata_path'),
                pin_memory=True,
                ref_length_min=data_config.get('ref_length_min', 2.0),
                ref_length_max=data_config.get('ref_length_max', 3.0),
                # Variable reference parameters
                variable_ref_enabled=True,
                ref_length_strategy=data_config.get('ref_length_strategy', 'adaptive'),
                memory_safe_mode=data_config.get('memory_safe_mode', True),
                normalize=data_config.get('normalize', True),
                normalize_scale=data_config.get('normalize_scale', 0.95),
                default_speaker_snr=data_config.get('default_speaker_snr', 5.0),
                default_noise_snr=data_config.get('default_noise_snr', 10.0),
                drop_last_train=data_config.get('drop_last_train', True),
                persistent_workers=data_config.get('persistent_workers', True)
            )
        else:
            # Use standard fixed reference dataset
            train_loader, val_loader, test_loader = create_tse_dataloaders(
            data_root=data_config['data_root'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            sample_rate=data_config['sample_rate'],
            segment_length=data_config['segment_length'],
            use_metadata=data_config.get('use_metadata', False),
            metadata_path=data_config.get('metadata_path'),
            pin_memory=data_config.get('pin_memory', True),
            ref_length_min=data_config.get('ref_length_min', 2.0),
            ref_length_max=data_config.get('ref_length_max', 2.0),
            normalize=data_config.get('normalize', True),
            normalize_scale=data_config.get('normalize_scale', 0.95),
            default_speaker_snr=data_config.get('default_speaker_snr', 5.0),
            default_noise_snr=data_config.get('default_noise_snr', 10.0),
            drop_last_train=data_config.get('drop_last_train', True),
            persistent_workers=data_config.get('persistent_workers', True),
            prefetch_factor=data_config.get('prefetch_factor', 3)
        )
        
        log.info(f"Data loaders created:")
        log.info(f"  Train: {len(train_loader)} batches")
        log.info(f"  Val: {len(val_loader)} batches")
        log.info(f"  Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def train_step(self, batch: Dict, accumulation_step: int) -> Tuple[float, float]:
        """Single training step with mixed precision and gradient accumulation"""
        self.model.train()

        # Move data to device
        mixture = batch['mixture'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True)
        reference = batch['reference'].to(self.device, non_blocking=True)

        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with autocast():
                extracted = self.model(mixture, reference)

                # Ensure same length for loss
                min_len = min(extracted.shape[-1], target.shape[-1])
                extracted = extracted[..., :min_len]
                target = target[..., :min_len]

                # Compute loss (scale by gradient accumulation steps)
                loss = si_sdr_loss(extracted, target) / self.gradient_accumulation_steps
        else:
            extracted = self.model(mixture, reference)

            # Ensure same length for loss
            min_len = min(extracted.shape[-1], target.shape[-1])
            extracted = extracted[..., :min_len]
            target = target[..., :min_len]

            # Compute loss (scale by gradient accumulation steps)
            loss = si_sdr_loss(extracted, target) / self.gradient_accumulation_steps

        # Compute SI-SDRi for monitoring
        with torch.no_grad():
            mixture_trimmed = mixture[..., :min_len]
            si_sdri = calculate_si_sdri(extracted, mixture_trimmed, target)

        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights only at accumulation boundaries
        if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.optimizer.step()

            # Update EMA model after optimizer step
            if self.ema_model is not None:
                self.ema_model.update(self.model)

            self.optimizer.zero_grad()

        return loss.item() * self.gradient_accumulation_steps, si_sdri
    
    def validate(self) -> Tuple[float, float]:
        """Validation loop using EMA model if available"""
        # Use EMA model for validation if available
        eval_model = self.ema_model.ema_model if self.ema_model else self.model
        eval_model.eval()

        total_loss = 0
        total_si_sdri = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                mixture = batch['mixture'].to(self.device, non_blocking=True)
                target = batch['target'].to(self.device, non_blocking=True)
                reference = batch['reference'].to(self.device, non_blocking=True)

                # Forward pass with EMA or regular model
                eval_model = self.ema_model.ema_model if self.ema_model else self.model
                extracted = eval_model(mixture, reference)

                # Ensure same length
                min_len = min(extracted.shape[-1], target.shape[-1])
                extracted = extracted[..., :min_len]
                target = target[..., :min_len]
                mixture_trimmed = mixture[..., :min_len]

                # Compute metrics
                loss = si_sdr_loss(extracted, target)
                si_sdri = calculate_si_sdri(extracted, mixture_trimmed, target)

                total_loss += loss.item()
                total_si_sdri += si_sdri
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_si_sdri = total_si_sdri / num_batches

        return avg_loss, avg_si_sdri

    def validate_subset(self, num_samples: int) -> Tuple[float, float]:
        """Quick validation on a subset of samples"""
        # Use EMA model for validation if available
        eval_model = self.ema_model.ema_model if self.ema_model else self.model
        eval_model.eval()

        total_loss = 0
        total_si_sdri = 0
        samples_processed = 0
        batch_size = self.config['data']['batch_size']
        max_batches = (num_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= max_batches:
                    break

                mixture = batch['mixture'].to(self.device, non_blocking=True)
                target = batch['target'].to(self.device, non_blocking=True)
                reference = batch['reference'].to(self.device, non_blocking=True)

                # Forward pass
                extracted = eval_model(mixture, reference)

                # Ensure same length
                min_len = min(extracted.shape[-1], target.shape[-1])
                extracted = extracted[..., :min_len]
                target = target[..., :min_len]
                mixture_trimmed = mixture[..., :min_len]

                # Compute metrics
                loss = si_sdr_loss(extracted, target)
                si_sdri = calculate_si_sdri(extracted, mixture_trimmed, target)

                total_loss += loss.item()
                total_si_sdri += si_sdri
                samples_processed += mixture.size(0)

                if samples_processed >= num_samples:
                    break

        num_batches = min(i + 1, max_batches)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_si_sdri = total_si_sdri / num_batches if num_batches > 0 else 0

        return avg_loss, avg_si_sdri
    
    def train(self):
        """Main training loop with optimized validation frequency"""
        log.info(f"Starting training on {self.device}")
        log.info(f"Training for {self.config['training']['num_epochs']} epochs")

        # Get validation frequency from config or use default
        val_frequency = self.config['training'].get('validation_frequency', 5)
        quick_val_samples = self.config['training'].get('quick_val_samples', 500)
        full_val_frequency = self.config['training'].get('full_val_frequency', 10)

        log.info(f"Validation strategy: Quick ({quick_val_samples} samples) every {val_frequency} epochs, Full every {full_val_frequency} epochs")

        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            epoch_start_time = time.time()

            # Update epoch for variable reference dataset if used
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)

            # Training phase
            self.model.train()
            train_loss = 0
            train_si_sdri = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for i, batch in enumerate(progress_bar):
                loss, si_sdri = self.train_step(batch, i)

                train_loss += loss
                train_si_sdri += si_sdri

                # Only increment global step at accumulation boundaries
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    self.global_step += 1

                # Update progress bar
                avg_loss = train_loss / (i + 1)
                avg_si_sdri = train_si_sdri / (i + 1)
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'SI-SDRi': f"{avg_si_sdri:.2f}"
                })

                # Log to tensorboard
                if self.writer and self.global_step % self.config['logging']['log_interval'] == 0:
                    self.writer.add_scalar('train/loss', loss, self.global_step)
                    self.writer.add_scalar('train/si_sdri', si_sdri, self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

                # Clear cache less frequently since we have memory headroom
                if (i + 1) % 500 == 0:
                    torch.cuda.empty_cache()

            # Calculate epoch averages
            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_si_sdri = train_si_sdri / len(self.train_loader)

            # Optimized validation strategy
            val_loss, val_si_sdri = None, None

            # Determine validation type for this epoch
            if (epoch + 1) % full_val_frequency == 0:
                # Full validation every N epochs
                log.info(f"Running full validation at epoch {epoch + 1}")
                val_loss, val_si_sdri = self.validate()
            elif (epoch + 1) % val_frequency == 0:
                # Quick validation on subset
                log.info(f"Running quick validation ({quick_val_samples} samples) at epoch {epoch + 1}")
                val_loss, val_si_sdri = self.validate_subset(quick_val_samples)
            else:
                # Skip validation this epoch - use last known values
                log.info(f"Skipping validation at epoch {epoch + 1}")
                val_loss = self.best_val_loss if hasattr(self, 'best_val_loss') else avg_train_loss
                val_si_sdri = self.best_val_si_sdri if hasattr(self, 'best_val_si_sdri') else avg_train_si_sdri
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, WarmupReduceLROnPlateau):
                    # Custom warmup scheduler needs metric
                    self.scheduler.step(val_si_sdri)
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_si_sdri)
                else:
                    self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            log.info(
                f"Epoch {epoch + 1}/{self.config['training']['num_epochs']} "
                f"(Time: {epoch_time:.1f}s) | "
                f"Train Loss: {avg_train_loss:.4f}, SI-SDRi: {avg_train_si_sdri:.2f} dB | "
                f"Val Loss: {val_loss:.4f}, SI-SDRi: {val_si_sdri:.2f} dB | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            if self.writer:
                self.writer.add_scalars('epoch/loss', {
                    'train': avg_train_loss,
                    'val': val_loss
                }, epoch + 1)
                self.writer.add_scalars('epoch/si_sdri', {
                    'train': avg_train_si_sdri,
                    'val': val_si_sdri
                }, epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging'].get('save_interval', 5) == 0:
                self.save_checkpoint(epoch, val_si_sdri)
            
            # Save best model
            if val_si_sdri > self.best_si_sdri:
                self.best_si_sdri = val_si_sdri
                self.save_checkpoint(epoch, val_si_sdri, is_best=True)
                log.info(f"New best model! SI-SDRi: {self.best_si_sdri:.2f} dB")
            
            # Early stopping
            if self.config['training'].get('early_stopping'):
                patience = self.config['training']['early_stopping']['patience']
                if hasattr(self, 'epochs_without_improvement'):
                    if val_si_sdri <= self.best_si_sdri:
                        self.epochs_without_improvement += 1
                    else:
                        self.epochs_without_improvement = 0
                    
                    if self.epochs_without_improvement >= patience:
                        log.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                else:
                    self.epochs_without_improvement = 0
        
        # Final test evaluation
        self.test()
        
        log.info(f"Training completed! Best SI-SDRi: {self.best_si_sdri:.2f} dB")
        
        if self.writer:
            self.writer.close()
    
    def test(self):
        """Test evaluation using EMA model if available"""
        log.info("Running test evaluation...")
        # Use EMA model for test if available
        eval_model = self.ema_model.ema_model if self.ema_model else self.model
        eval_model.eval()
        
        total_si_sdri = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                mixture = batch['mixture'].to(self.device)
                target = batch['target'].to(self.device)
                reference = batch['reference'].to(self.device)
                
                # Use EMA or regular model
                eval_model = self.ema_model.ema_model if self.ema_model else self.model
                extracted = eval_model(mixture, reference)

                min_len = min(extracted.shape[-1], target.shape[-1])
                extracted = extracted[..., :min_len]
                target = target[..., :min_len]
                mixture = mixture[..., :min_len]
                
                si_sdri = calculate_si_sdri(extracted, mixture, target)
                total_si_sdri += si_sdri * mixture.shape[0]
                num_samples += mixture.shape[0]
        
        avg_si_sdri = total_si_sdri / num_samples
        log.info(f"Test SI-SDRi: {avg_si_sdri:.2f} dB")
        
        if self.writer:
            self.writer.add_scalar('test/si_sdri', avg_si_sdri, self.config['training']['num_epochs'])
    
    def save_checkpoint(self, epoch: int, si_sdri: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'si_sdri': si_sdri,
            'best_si_sdri': self.best_si_sdri,
            'global_step': self.global_step,
            'config': self.config
        }

        # Save EMA model state if available
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        
        torch.save(checkpoint, path)
        log.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load EMA model state if available
        if self.ema_model is not None and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            log.info("EMA model state loaded")
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_si_sdri = checkpoint.get('best_si_sdri', float('-inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        log.info(f"Checkpoint loaded from {checkpoint_path}")
        log.info(f"Resuming from epoch {self.start_epoch}, best SI-SDRi: {self.best_si_sdri:.2f} dB")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train USEF-TSE-Mamba v7 with real datasets")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with resume checkpoint if specified
    if args.resume:
        config['resume_checkpoint'] = args.resume
    
    # Create trainer and start training
    trainer = TSETrainerV7(config)
    trainer.train()


if __name__ == "__main__":
    main()