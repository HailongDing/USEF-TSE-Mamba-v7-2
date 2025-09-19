"""
USEF-TSE-Mamba v7 Enhanced Model
Architecture improvements:
1. Batch normalization after conv layers
2. SiLU activation instead of ReLU
3. Cross-attention only at bottleneck (more efficient)
4. Gated skip connections for better gradient flow
5. Layer normalization in critical paths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import (
    create_block,
    _init_weights,
)
from torch.nn import Conv1d, ConvTranspose1d, ModuleList
from functools import partial
import math

# Modern activation functions
activations = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,  # Swish activation - better gradient flow
    "gelu": nn.GELU,
    "mish": nn.Mish,
    "prelu": nn.PReLU
}


class GatedSkipConnection(nn.Module):
    """Gated skip connection for better gradient flow and feature selection"""

    def __init__(self, dim):
        super(GatedSkipConnection, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        """
        Args:
            x: Current features (B, C, L)
            skip: Skip connection features (B, C, L)
        Returns:
            Gated combination of x and skip
        """
        B, C, L = x.shape

        # Permute to (B, L, C) for linear layers
        x_perm = x.permute(0, 2, 1)
        skip_perm = skip.permute(0, 2, 1)

        # Concatenate and compute gate
        combined = torch.cat([x_perm, skip_perm], dim=-1)
        gate_values = self.gate(combined)

        # Apply gate
        output = gate_values * skip_perm + (1 - gate_values) * x_perm

        # Permute back to (B, C, L)
        return output.permute(0, 2, 1)


class EnhancedDownSample(nn.Module):
    """Enhanced downsampling block with batch norm and better activations"""

    def __init__(
        self, features_in, features_out, kernel_size, stride, num_blocks, act_fn="silu"
    ):
        super(EnhancedDownSample, self).__init__()

        # Conv block with batch norm
        self.conv_block = nn.Sequential(
            Conv1d(
                in_channels=features_in,
                out_channels=features_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2  # Add padding for better boundary handling
            ),
            nn.BatchNorm1d(features_out),  # Batch normalization
            activations[act_fn]()
        )

        # Mamba layer for sequence modeling
        self.mamba_layer = MambaLayer(dim=features_out, num_blocks=num_blocks)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(features_out)

    def forward(self, x):
        # Apply convolution with batch norm
        x = self.conv_block(x)

        # Mamba processing
        x, res_forward = self.mamba_layer(x)

        # Apply layer norm (permute for LayerNorm)
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

        return x, res_forward


class EnhancedUpSample(nn.Module):
    """Enhanced upsampling block with gated skip connections"""

    def __init__(
        self, features_in, features_out, kernel_size, stride, num_blocks, act_fn="silu"
    ):
        super(EnhancedUpSample, self).__init__()

        # Transposed conv with batch norm
        self.convT_block = nn.Sequential(
            ConvTranspose1d(
                in_channels=features_in,
                out_channels=features_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2,
                output_padding=stride-1  # Ensure correct output size
            ),
            nn.BatchNorm1d(features_out),
            activations[act_fn]()
        )

        # Fusion convolution with batch norm
        self.conv1d_fusion = nn.Sequential(
            Conv1d(
                in_channels=features_in,  # After concatenation
                out_channels=features_out,
                kernel_size=1
            ),
            nn.BatchNorm1d(features_out),
            activations[act_fn]()
        )

        # Gated skip connection
        self.gated_skip = GatedSkipConnection(features_out)

        # Mamba layer
        self.mamba_layer = MambaLayer(
            dim=features_out, num_blocks=num_blocks, start_idx=0
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(features_out)

    def pad(self, x, match):
        """Pad x to match the size of match tensor"""
        if match.shape[2] > x.shape[2]:
            zero_pad = torch.zeros(
                x.shape[0], x.shape[1], match.shape[2] - x.shape[2],
                device=x.device, dtype=x.dtype
            )
            return torch.cat((x, zero_pad), dim=2)
        else:
            return x[:, :, :match.shape[2]]  # Trim if necessary

    def forward(self, x, skip):
        # Upsample
        x = self.convT_block(x)

        # Pad to match skip connection
        x = self.pad(x, match=skip)

        # Concatenate and fuse
        x = torch.cat([skip, x], dim=1)
        x = self.conv1d_fusion(x)

        # Apply gated skip connection
        x_before_mamba = x.clone()
        x, _ = self.mamba_layer(x)
        x = self.gated_skip(x, x_before_mamba)

        # Layer norm
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)

        return x


class MambaLayer(nn.Module):
    """Mamba layer with bidirectional processing"""

    def __init__(self, dim, num_blocks, start_idx=0):
        super(MambaLayer, self).__init__()
        self.forward_mamba_block = MambaBlock(
            dim=dim, num_blocks=num_blocks, start_idx=start_idx
        )

    def forward(self, x, res_for=None):
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        x_forward = x.clone()

        x_forward, res_forward = self.forward_mamba_block(x_forward, res_for)
        x_forward = x_forward + res_forward  # Residual connection

        x = x_forward.clone()
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

        return x, res_forward


class MambaBlock(nn.Module):
    """Mamba block with multiple layers"""

    def __init__(self, dim, start_idx=0, num_blocks=3):
        super(MambaBlock, self).__init__()

        self.mamba_blocks = ModuleList(
            [
                create_block(d_model=dim, layer_idx=i)
                for i in range(start_idx, start_idx + num_blocks)
            ]
        )

        self.apply(partial(_init_weights, n_layer=num_blocks))

    def forward(self, x, residual=None):
        for block in self.mamba_blocks:
            x, residual = block(x, residual)

        return x, residual


class EfficientCrossAttention(nn.Module):
    """
    Efficient Cross-Attention module
    - Only applied at bottleneck for efficiency
    - Uses Flash Attention pattern for speed
    - Includes positional encoding
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, max_len=5000):
        super(EfficientCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Multi-head projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Layer norms for stability
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.01)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key_value):
        """
        Args:
            query: Mixed speech features (B, C, L_mix)
            key_value: Reference speech features (B, C, L_ref)
        Returns:
            Frame-level speaker features (B, C, L_mix)
        """
        B, C, L_mix = query.shape
        _, _, L_ref = key_value.shape

        # Transpose to (B, L, C) for attention
        query = query.transpose(1, 2)  # (B, L_mix, C)
        key_value = key_value.transpose(1, 2)  # (B, L_ref, C)

        # Add positional encoding
        if L_mix <= self.pos_encoding.shape[1]:
            query = query + self.pos_encoding[:, :L_mix, :]
        if L_ref <= self.pos_encoding.shape[1]:
            key_value = key_value + self.pos_encoding[:, :L_ref, :]

        # Normalize before projection for stability
        query = self.q_norm(query)
        key_value = self.k_norm(key_value)

        # Project to Q, K, V
        Q = self.q_proj(query)  # (B, L_mix, C)
        K = self.k_proj(key_value)  # (B, L_ref, C)
        V = self.v_proj(key_value)  # (B, L_ref, C)

        # Reshape for multi-head attention
        Q = Q.view(B, L_mix, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_mix, D)
        K = K.view(B, L_ref, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_ref, D)
        V = V.view(B, L_ref, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_ref, D)

        # Scaled dot-product attention with numerical stability
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L_mix, L_ref)

        # Clamp scores for stability
        scores = torch.clamp(scores, min=-50, max=50)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, H, L_mix, D)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_mix, C)  # (B, L_mix, C)

        # Output projection
        output = self.out_proj(attn_output)  # (B, L_mix, C)

        # Residual connection
        output = output + query

        # Transpose back to (B, C, L_mix)
        output = output.transpose(1, 2)

        return output


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive feature fusion with learned weights
    More sophisticated than simple weighted addition
    """

    def __init__(self, embed_dim):
        super(AdaptiveFeatureFusion, self).__init__()

        # Adaptive fusion network
        self.fusion_net = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim)
        )

        # Attention weights for fusion
        self.attention = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(embed_dim, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, mixture_features, speaker_features):
        """
        Args:
            mixture_features: (B, C, L)
            speaker_features: (B, C, L)
        Returns:
            Fused features (B, C, L)
        """
        # Concatenate features
        concat_features = torch.cat([mixture_features, speaker_features], dim=1)

        # Compute attention weights
        weights = self.attention(concat_features)  # (B, 2, L)

        # Apply weighted combination
        weighted_mixture = mixture_features * weights[:, 0:1, :]
        weighted_speaker = speaker_features * weights[:, 1:2, :]
        fused = weighted_mixture + weighted_speaker

        # Apply fusion network for non-linear combination
        fusion_output = self.fusion_net(concat_features)

        # Combine linear and non-linear fusion
        output = fused + fusion_output

        # Layer norm for stability
        output = output.permute(0, 2, 1)  # (B, L, C)
        output = self.layer_norm(output)
        output = output.permute(0, 2, 1)  # (B, C, L)

        return output


class USEF_TSE_Mamba_v7_Enhanced(nn.Module):
    """
    Enhanced USEF-TSE Mamba v7 with architectural improvements

    Key improvements:
    1. Batch normalization throughout
    2. SiLU activation for better gradient flow
    3. Cross-attention only at bottleneck (more efficient)
    4. Gated skip connections
    5. Adaptive feature fusion
    6. Better numerical stability
    """

    def __init__(
        self,
        dim=128,
        kernel_sizes=[16, 16, 16],
        strides=[2, 2, 2],
        num_blocks=6,
        act_fn="silu",  # Default to SiLU
        num_heads=8,
        dropout=0.1,
        use_bottleneck_attention_only=True  # New parameter
    ):
        super(USEF_TSE_Mamba_v7_Enhanced, self).__init__()
        assert len(kernel_sizes) == len(strides)

        self.use_bottleneck_attention_only = use_bottleneck_attention_only

        # Enhanced encoder blocks with batch norm and SiLU
        self.encoder_blocks = ModuleList([
            EnhancedDownSample(
                features_in=1,
                features_out=dim,
                kernel_size=kernel_sizes[0],
                stride=strides[0],
                num_blocks=num_blocks,
                act_fn=act_fn
            ),
            EnhancedDownSample(
                features_in=dim,
                features_out=dim * 2,
                kernel_size=kernel_sizes[1],
                stride=strides[1],
                num_blocks=num_blocks,
                act_fn=act_fn
            ),
            EnhancedDownSample(
                features_in=dim * 2,
                features_out=dim * 4,
                kernel_size=kernel_sizes[2],
                stride=strides[2],
                num_blocks=num_blocks,
                act_fn=act_fn
            ),
        ])

        if use_bottleneck_attention_only:
            # Single efficient cross-attention at bottleneck
            self.bottleneck_attention = EfficientCrossAttention(
                dim * 4, num_heads=num_heads, dropout=dropout
            )
            self.bottleneck_fusion = AdaptiveFeatureFusion(dim * 4)
        else:
            # Original: Cross-attention at each level
            self.cross_attention_modules = ModuleList([
                EfficientCrossAttention(dim, num_heads=num_heads//2, dropout=dropout),
                EfficientCrossAttention(dim * 2, num_heads=num_heads, dropout=dropout),
                EfficientCrossAttention(dim * 4, num_heads=num_heads, dropout=dropout),
            ])

            self.fusion_modules = ModuleList([
                AdaptiveFeatureFusion(dim),
                AdaptiveFeatureFusion(dim * 2),
                AdaptiveFeatureFusion(dim * 4),
            ])

        # Enhanced decoder blocks with gated skip connections
        self.decoder_blocks = ModuleList([
            EnhancedUpSample(
                features_in=dim * 4,
                features_out=dim * 2,
                kernel_size=kernel_sizes[2],
                stride=strides[2],
                num_blocks=num_blocks,
                act_fn=act_fn
            ),
            EnhancedUpSample(
                features_in=dim * 2,
                features_out=dim,
                kernel_size=kernel_sizes[1],
                stride=strides[1],
                num_blocks=num_blocks,
                act_fn=act_fn
            ),
        ])

        # Final decoder with batch norm
        self.final_decoder = nn.Sequential(
            ConvTranspose1d(
                in_channels=dim,
                out_channels=dim//2,
                kernel_size=kernel_sizes[0],
                stride=strides[0],
                padding=kernel_sizes[0]//2,
                output_padding=strides[0]-1
            ),
            nn.BatchNorm1d(dim//2),
            activations[act_fn](),
            nn.Conv1d(dim//2, 1, kernel_size=1)  # Final projection to single channel
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Custom weight initialization"""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def encode_with_sharing(self, x):
        """
        Encode input using shared encoder blocks
        Args:
            x: Input audio (B, 1, T)
        Returns:
            List of encoded features at each level
        """
        features = []
        for encoder in self.encoder_blocks:
            x, _ = encoder(x)
            features.append(x)
        return features

    def forward(self, mixture, reference):
        """
        Args:
            mixture: Mixed audio with multiple speakers (B, 1, T) or (B, T)
            reference: Reference audio from target speaker (B, 1, T_ref) or (B, T_ref)

        Returns:
            extracted: Extracted target speaker audio (B, 1, T)
        """
        # Ensure correct input shapes
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)
        if reference.dim() == 2:
            reference = reference.unsqueeze(1)

        # Encode both mixture and reference using shared encoders
        mixture_features = self.encode_with_sharing(mixture)
        reference_features = self.encode_with_sharing(reference)

        if self.use_bottleneck_attention_only:
            # Efficient: Only apply attention at bottleneck
            skips = []

            # Store skip connections from first two levels
            for i in range(2):
                skips.append(mixture_features[i])

            # Apply cross-attention and fusion only at bottleneck
            bottleneck_mix = mixture_features[-1]
            bottleneck_ref = reference_features[-1]

            speaker_features = self.bottleneck_attention(bottleneck_mix, bottleneck_ref)
            x = self.bottleneck_fusion(bottleneck_mix, speaker_features)

        else:
            # Original: Apply attention at each level
            fused_features = []
            skips = []

            for i, (mix_feat, ref_feat) in enumerate(zip(mixture_features, reference_features)):
                # Extract frame-level speaker features via cross-attention
                speaker_features = self.cross_attention_modules[i](mix_feat, ref_feat)

                # Fuse mixture features with speaker features
                fused = self.fusion_modules[i](mix_feat, speaker_features)
                fused_features.append(fused)
                skips.append(fused)

            # Use the deepest fused features for decoding
            x = fused_features[-1]

            # Remove bottleneck from skips
            skips.pop()

        # Decode with skip connections
        for decoder in self.decoder_blocks:
            skip = skips.pop()
            x = decoder(x, skip)

        # Final output
        extracted = self.final_decoder(x)

        return extracted


def compare_models():
    """Compare original and enhanced models"""
    import torchinfo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model parameters
    dim = 128
    num_blocks = 6
    kernel_sizes = [16, 16, 16]
    strides = [2, 2, 2]
    batch_size = 2

    # Create enhanced model
    model = USEF_TSE_Mamba_v7_Enhanced(
        dim=dim,
        num_blocks=num_blocks,
        kernel_sizes=kernel_sizes,
        strides=strides,
        act_fn="silu",
        num_heads=8,
        dropout=0.1,
        use_bottleneck_attention_only=True
    ).to(device)

    print("\n=== Enhanced USEF-TSE-Mamba v7 Model ===")
    print("Key Improvements:")
    print("✓ Batch normalization for training stability")
    print("✓ SiLU activation for better gradient flow")
    print("✓ Efficient bottleneck-only cross-attention")
    print("✓ Gated skip connections")
    print("✓ Adaptive feature fusion")
    print("✓ Positional encoding in attention")
    print("✓ Better numerical stability\n")

    # Test forward pass
    mixture = torch.randn(batch_size, 1, 40000).to(device)  # 5 seconds at 8kHz
    reference = torch.randn(batch_size, 1, 24000).to(device)  # 3 seconds

    model.eval()
    with torch.no_grad():
        extracted = model(mixture, reference)

    print(f"Input shapes:")
    print(f"  Mixture: {mixture.shape}")
    print(f"  Reference: {reference.shape}")
    print(f"Output shape:")
    print(f"  Extracted: {extracted.shape}")

    # Model summary
    print("\n=== Model Summary ===")
    torchinfo.summary(
        model,
        input_size=[(batch_size, 1, 40000), (batch_size, 1, 24000)],
        device=device,
        verbose=0
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


if __name__ == "__main__":
    compare_models()