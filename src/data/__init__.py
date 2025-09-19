# USEF-TSE-Mamba v7 Data Processing
# Note: Original versions have been archived
# Using optimized cached and augmented versions
from .tse_dataset_v7_cached import TSEDatasetV7Cached, create_tse_dataloaders_cached
from .tse_dataset_v7_augmented import TSEDatasetV7Augmented, create_augmented_dataloaders
from .generate_tse_dataset_v7_parallel import TSEDatasetGeneratorV7Parallel

# Aliases for backward compatibility
TSEDatasetV7 = TSEDatasetV7Cached
create_tse_dataloaders = create_tse_dataloaders_cached
TSEDatasetGeneratorV7 = TSEDatasetGeneratorV7Parallel

__all__ = [
    'TSEDatasetV7Cached',
    'TSEDatasetV7Augmented',
    'create_tse_dataloaders_cached',
    'create_augmented_dataloaders',
    'TSEDatasetGeneratorV7Parallel',
    # Backward compatibility
    'TSEDatasetV7',
    'create_tse_dataloaders',
    'TSEDatasetGeneratorV7'
]