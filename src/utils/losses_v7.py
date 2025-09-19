import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


def si_sdr_loss(estimated, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.
    
    Args:
        estimated: Estimated signal (B, T) or (B, 1, T)
        target: Target signal (B, T) or (B, 1, T)
        eps: Small value to avoid division by zero
    
    Returns:
        SI-SDR loss (scalar)
    """
    if estimated.dim() == 3:
        estimated = estimated.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    # Zero mean
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    # Compute SI-SDR
    dot = torch.sum(estimated * target, dim=-1, keepdim=True)
    target_norm = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    
    s_target = dot * target / target_norm
    e_noise = estimated - s_target
    
    si_sdr = 10 * torch.log10(
        torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + eps) + eps
    )
    
    return -si_sdr.mean()  # Negative because we want to maximize SI-SDR


def pit_loss(estimated_sources, target_sources, loss_fn=si_sdr_loss):
    """
    Permutation Invariant Training (PIT) loss.
    
    Args:
        estimated_sources: List of estimated sources, each (B, 1, T)
        target_sources: Tensor of target sources (B, num_sources, T)
        loss_fn: Loss function to use for each source pair
    
    Returns:
        PIT loss (scalar)
    """
    batch_size = target_sources.shape[0]
    num_sources = target_sources.shape[1]
    
    # Convert list to tensor if needed
    if isinstance(estimated_sources, list):
        estimated_sources = torch.stack([s.squeeze(1) for s in estimated_sources], dim=1)
    
    if estimated_sources.dim() == 3 and estimated_sources.shape[1] == 1:
        # If single source extraction, replicate for comparison
        estimated_sources = estimated_sources.expand(-1, num_sources, -1)
    
    # Compute loss for all permutations
    losses = []
    for perm in permutations(range(num_sources)):
        loss = 0
        for i, j in enumerate(perm):
            loss += loss_fn(estimated_sources[:, i], target_sources[:, j])
        losses.append(loss)
    
    # Return minimum loss across permutations
    return min(losses)


class USEF_TSE_Loss(nn.Module):
    """
    Combined loss for USEF-TSE training.
    
    The loss consists of:
    1. L_tse: Target speaker extraction loss (SI-SDR on extracted speech)
    2. L_aux: Auxiliary network loss (PIT loss on all speaker outputs)
    """
    def __init__(self, alpha=0.5, beta=0.5, use_pit=True):
        super(USEF_TSE_Loss, self).__init__()
        self.alpha = alpha  # Weight for TSE loss
        self.beta = beta    # Weight for auxiliary loss
        self.use_pit = use_pit
    
    def forward(self, extracted, aux_outputs, target_sources, speaker_weights=None, target_idx=None):
        """
        Args:
            extracted: Extracted target speech (B, 1, T)
            aux_outputs: List of auxiliary network outputs for each speaker
            target_sources: All source signals (B, num_sources, T)
            speaker_weights: Optional speaker selection weights (B, num_speakers)
            target_idx: Optional target speaker indices (B,)
        
        Returns:
            total_loss, tse_loss, aux_loss
        """
        batch_size = extracted.shape[0]
        
        # TSE Loss: SI-SDR between extracted and target
        if target_idx is not None:
            # Use specified target
            target_speech = []
            for b in range(batch_size):
                idx = target_idx[b].item() if torch.is_tensor(target_idx) else target_idx
                target_speech.append(target_sources[b, idx:idx+1])
            target_speech = torch.cat(target_speech, dim=0)
        else:
            # Use best matching target (oracle selection for training)
            losses = []
            for i in range(target_sources.shape[1]):
                loss = si_sdr_loss(extracted, target_sources[:, i])
                losses.append(loss)
            min_loss_idx = torch.argmin(torch.stack(losses))
            target_speech = target_sources[:, min_loss_idx]
        
        tse_loss = si_sdr_loss(extracted, target_speech)
        
        # Auxiliary Loss: PIT loss for all speaker outputs
        if self.use_pit:
            aux_loss = pit_loss(aux_outputs, target_sources)
        else:
            # Simple MSE loss for auxiliary outputs
            aux_loss = 0
            for i, aux_out in enumerate(aux_outputs):
                if i < target_sources.shape[1]:
                    aux_loss += F.mse_loss(aux_out, target_sources[:, i:i+1])
            aux_loss = aux_loss / len(aux_outputs)
        
        # Speaker selection loss (optional)
        selection_loss = 0
        if speaker_weights is not None and target_idx is not None:
            # Cross-entropy loss for speaker selection
            if not torch.is_tensor(target_idx):
                target_idx = torch.tensor([target_idx] * batch_size).to(speaker_weights.device)
            selection_loss = F.cross_entropy(speaker_weights, target_idx)
        
        # Combined loss
        total_loss = self.alpha * tse_loss + self.beta * aux_loss + 0.1 * selection_loss
        
        return total_loss, tse_loss, aux_loss


def calculate_si_sdri(estimated, mixture, target):
    """
    Calculate SI-SDR improvement (SI-SDRi).
    
    Args:
        estimated: Estimated signal
        mixture: Mixed signal
        target: Target clean signal
    
    Returns:
        SI-SDR improvement in dB
    """
    si_sdr_est = -si_sdr_loss(estimated, target)
    si_sdr_mix = -si_sdr_loss(mixture, target)
    return (si_sdr_est - si_sdr_mix).item()


def reorder_sources(estimated_sources, target_sources):
    """
    Reorder estimated sources to match target sources (solving permutation problem).
    
    Args:
        estimated_sources: Estimated sources (B, num_sources, T)
        target_sources: Target sources (B, num_sources, T)
    
    Returns:
        Reordered estimated sources
    """
    batch_size, num_sources, length = estimated_sources.shape
    reordered = torch.zeros_like(estimated_sources)
    
    for b in range(batch_size):
        # Compute SI-SDR for all pairs
        sdr_matrix = torch.zeros(num_sources, num_sources)
        for i in range(num_sources):
            for j in range(num_sources):
                sdr_matrix[i, j] = -si_sdr_loss(
                    estimated_sources[b, i], 
                    target_sources[b, j]
                )
        
        # Find best permutation (Hungarian algorithm would be better, but this works for 2 sources)
        if num_sources == 2:
            if sdr_matrix[0, 0] + sdr_matrix[1, 1] >= sdr_matrix[0, 1] + sdr_matrix[1, 0]:
                reordered[b] = estimated_sources[b]
            else:
                reordered[b, 0] = estimated_sources[b, 1]
                reordered[b, 1] = estimated_sources[b, 0]
        else:
            # For more sources, use greedy assignment
            used_targets = []
            for i in range(num_sources):
                available = [j for j in range(num_sources) if j not in used_targets]
                best_j = max(available, key=lambda j: sdr_matrix[i, j])
                reordered[b, best_j] = estimated_sources[b, i]
                used_targets.append(best_j)
    
    return reordered


if __name__ == "__main__":
    # Test the losses
    batch_size = 2
    num_sources = 2
    length = 8000
    
    # Create dummy data
    extracted = torch.randn(batch_size, 1, length)
    aux_outputs = [torch.randn(batch_size, 1, length) for _ in range(num_sources)]
    target_sources = torch.randn(batch_size, num_sources, length)
    speaker_weights = torch.softmax(torch.randn(batch_size, num_sources), dim=-1)
    target_idx = torch.randint(0, num_sources, (batch_size,))
    
    # Test USEF-TSE loss
    loss_fn = USEF_TSE_Loss(alpha=0.5, beta=0.5)
    total_loss, tse_loss, aux_loss = loss_fn(
        extracted, aux_outputs, target_sources, speaker_weights, target_idx
    )
    
    print(f"Total loss: {total_loss:.4f}")
    print(f"TSE loss: {tse_loss:.4f}")
    print(f"Auxiliary loss: {aux_loss:.4f}")
    
    # Test SI-SDRi
    mixture = target_sources.sum(dim=1, keepdim=True)
    si_sdri = calculate_si_sdri(extracted, mixture, target_sources[:, 0:1])
    print(f"SI-SDRi: {si_sdri:.2f} dB")