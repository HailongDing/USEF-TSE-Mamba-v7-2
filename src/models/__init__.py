# USEF-TSE-Mamba v7 Models
# Note: Original USEF_TSE_Mamba_v7 has been archived
# Only the enhanced version is available
from .USEF_TSE_Mamba_v7_enhanced import USEF_TSE_Mamba_v7_Enhanced

# For backward compatibility, alias the enhanced model
USEF_TSE_Mamba_v7 = USEF_TSE_Mamba_v7_Enhanced

__all__ = ['USEF_TSE_Mamba_v7_Enhanced', 'USEF_TSE_Mamba_v7']