"""
Exponential Moving Average (EMA) for model weights
Helps prevent overfitting by maintaining a smoothed version of model parameters
"""

import torch
import torch.nn as nn
from typing import Optional
import copy


class ModelEMA:
    """
    Exponential Moving Average of model parameters

    Args:
        model: PyTorch model to track
        decay: EMA decay rate (0.999 means slow update, 0.9 means fast update)
        device: Device to store EMA model
        update_every: Update EMA every N steps (for efficiency)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
        update_every: int = 10
    ):
        # Create EMA model as a deep copy
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Move to device if specified
        if device is not None:
            self.ema_model = self.ema_model.to(device)

        self.decay = decay
        self.update_every = update_every
        self.updates = 0

        # Disable gradient computation for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

        # Initialize EMA weights to match original model
        self.ema_model.load_state_dict(model.state_dict())

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA parameters

        Args:
            model: Current model with updated parameters
        """
        self.updates += 1

        # Only update every N steps for efficiency
        if self.updates % self.update_every != 0:
            return

        # Compute EMA decay (increases from 0 to decay as training progresses)
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))

        # Update EMA parameters
        ema_params = self.ema_model.state_dict()
        model_params = model.state_dict()

        for key in ema_params.keys():
            if key in model_params:
                # Skip non-float tensors (e.g., BatchNorm's num_batches_tracked)
                if ema_params[key].dtype in [torch.float32, torch.float16, torch.float64]:
                    # EMA update: ema = decay * ema + (1 - decay) * model
                    ema_params[key].mul_(decay).add_(model_params[key], alpha=1 - decay)
                else:
                    # For non-float parameters (like counts), just copy them
                    ema_params[key].copy_(model_params[key])

        self.ema_model.load_state_dict(ema_params)

    def forward(self, *args, **kwargs):
        """Forward pass using EMA model"""
        return self.ema_model(*args, **kwargs)

    def eval(self):
        """Set EMA model to eval mode"""
        self.ema_model.eval()

    def train(self):
        """EMA model should always be in eval mode"""
        self.ema_model.eval()

    def state_dict(self):
        """Get EMA model state dict"""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load EMA model state dict"""
        self.ema_model.load_state_dict(state_dict)

    def to(self, device):
        """Move EMA model to device"""
        self.ema_model = self.ema_model.to(device)
        return self

    @property
    def module(self):
        """Access underlying model (for compatibility)"""
        return self.ema_model