#!/bin/bash
# USEF-TSE-Mamba V7 Dataset Regeneration Script
# Optimized configuration: 30K train / 2K val / 1K test

echo "=========================================="
echo "USEF-TSE-Mamba V7 Dataset Regeneration"
echo "Optimized: 30K/2K/1K configuration"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "usef_tse" ]]; then
    echo -e "${YELLOW}Activating conda environment 'usef_tse'...${NC}"
    # Try different conda paths
    if [ -f "/home/shiqi/anaconda3/etc/profile.d/conda.sh" ]; then
        source /home/shiqi/anaconda3/etc/profile.d/conda.sh
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source $HOME/miniconda3/etc/profile.d/conda.sh
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source /opt/conda/etc/profile.d/conda.sh
    else
        echo -e "${YELLOW}Warning: Could not find conda.sh. Please run: conda activate usef_tse${NC}"
    fi
    conda activate usef_tse 2>/dev/null || echo -e "${YELLOW}Please activate manually: conda activate usef_tse${NC}"
fi

# Configuration
WORKERS=8
CONFIG_FILE="configs/dataset_config_v7_optimized.yaml"
# Only parallel generator is available (standard was archived)
GENERATOR_SCRIPT="src/data/generate_tse_dataset_v7_parallel.py"

# Check if required files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check if generator script exists
if [ ! -f "$GENERATOR_SCRIPT" ]; then
    echo -e "${RED}Error: Generator script not found: $GENERATOR_SCRIPT${NC}"
    exit 1
fi

# Ask user about worker count
echo ""
echo "Select parallel processing workers:"
echo "1) Fast (8 workers) - Uses more CPU/RAM"
echo "2) Moderate (4 workers) - Balanced"
echo "3) Conservative (2 workers) - Low resource usage"
echo -n "Enter choice [1-3]: "
read choice

# Set workers based on choice
if [ "$choice" == "1" ]; then
    WORKERS=8
    echo -e "${GREEN}Using 8 parallel workers (fastest)${NC}"
elif [ "$choice" == "2" ]; then
    WORKERS=4
    echo -e "${GREEN}Using 4 parallel workers (balanced)${NC}"
else
    WORKERS=2
    echo -e "${GREEN}Using 2 parallel workers (conservative)${NC}"
fi

# Check if librosa is installed
python -c "import librosa" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing librosa for better audio quality...${NC}"
    pip install librosa
fi

# Check disk space
echo ""
echo "Checking disk space..."
REQUIRED_GB=10
AVAILABLE_GB=$(df /data | awk 'NR==2 {print int($4/1048576)}')

if [ $AVAILABLE_GB -lt $REQUIRED_GB ]; then
    echo -e "${RED}Error: Insufficient disk space. Need ${REQUIRED_GB}GB, have ${AVAILABLE_GB}GB${NC}"
    exit 1
fi
echo -e "${GREEN}Disk space OK: ${AVAILABLE_GB}GB available${NC}"

# Show dataset configuration
echo ""
echo "Dataset configuration (from $CONFIG_FILE):"
echo "  Train:      30,000 samples (50 hours)"
echo "  Validation:  2,000 samples (3.3 hours)"
echo "  Test:        1,000 samples (1.7 hours)"
echo "  Total:      33,000 samples (55 hours)"
echo ""
echo "Audio configuration:"
echo "  Mixture duration:   6.0 seconds"
echo "  Reference duration: 4.0 seconds"
echo "  Sample rate:        8000 Hz"
echo ""

# Estimate time based on workers
if [ "$WORKERS" == "8" ]; then
    echo -e "${YELLOW}Estimated time: 45-60 minutes${NC}"
elif [ "$WORKERS" == "4" ]; then
    echo -e "${YELLOW}Estimated time: 60-90 minutes${NC}"
else
    echo -e "${YELLOW}Estimated time: 90-120 minutes${NC}"
fi

# Backup existing dataset if it exists
DATASET_DIR="/data/tse_audio_dataset_v7_optimized"
if [ -d "$DATASET_DIR" ]; then
    echo ""
    echo -e "${YELLOW}Existing dataset found at $DATASET_DIR${NC}"
    echo -n "Backup existing dataset? (y/n): "
    read backup

    if [ "$backup" == "y" ]; then
        BACKUP_DIR="${DATASET_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        echo "Creating backup at $BACKUP_DIR..."
        mv "$DATASET_DIR" "$BACKUP_DIR"
        echo -e "${GREEN}Backup created${NC}"
    else
        echo -n "Delete existing dataset? (y/n): "
        read delete
        if [ "$delete" == "y" ]; then
            rm -rf "$DATASET_DIR"
            echo -e "${GREEN}Existing dataset removed${NC}"
        else
            echo -e "${RED}Cannot proceed with existing dataset. Exiting.${NC}"
            exit 1
        fi
    fi
fi

# Final confirmation
echo ""
echo -e "${YELLOW}Ready to generate optimized dataset${NC}"
echo -n "Proceed? (y/n): "
read confirm

if [ "$confirm" != "y" ]; then
    echo "Generation cancelled"
    exit 0
fi

# Start generation
echo ""
echo "=========================================="
echo "Starting dataset generation..."
echo "=========================================="

START_TIME=$(date +%s)

# Run generation with parallel processing
python "$GENERATOR_SCRIPT" \
    --config "$CONFIG_FILE" \
    --split all \
    --workers $WORKERS

# Check if generation was successful
if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))

    echo ""
    echo "=========================================="
    echo -e "${GREEN}Dataset generation completed successfully!${NC}"
    echo "Time taken: ${ELAPSED_MIN} minutes"
    echo "=========================================="

    # Show dataset statistics
    echo ""
    echo "Dataset statistics:"
    ls -lh "$DATASET_DIR"/train/mixture/*.wav 2>/dev/null | wc -l | xargs echo "  Train samples:"
    ls -lh "$DATASET_DIR"/validation/mixture/*.wav 2>/dev/null | wc -l | xargs echo "  Validation samples:"
    ls -lh "$DATASET_DIR"/test/mixture/*.wav 2>/dev/null | wc -l | xargs echo "  Test samples:"

    # Show disk usage
    echo ""
    echo "Disk usage:"
    du -sh "$DATASET_DIR" 2>/dev/null

    # Update training config to use new dataset
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. The optimized dataset is ready at: $DATASET_DIR"
    echo "2. To train with the new dataset:"
    echo "   python src/train_v7.py --config configs/config_v7_optimized.yaml"
    echo ""
    echo "Benefits of the optimized dataset:"
    echo "  • 50% more training data (better convergence)"
    echo "  • 60% less validation overhead (faster epochs)"
    echo "  • Better speaker diversity"
    echo "  • Expected SI-SDRi: 11-13 dB"

else
    echo ""
    echo "=========================================="
    echo -e "${RED}Dataset generation failed!${NC}"
    echo "Check the logs for errors"
    echo "=========================================="
    exit 1
fi