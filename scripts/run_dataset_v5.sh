#!/bin/bash

# =====================================================
#  TSE Dataset V5 Generation Script
#  USEF-TSE-Mamba Project
#  
#  DEFAULT: Auto-cleans previous data for fresh generation
#  Use --keep-existing to preserve existing files
# =====================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}=====================================================${NC}"
echo -e "${BLUE}       TSE Dataset V5 Generation Pipeline${NC}"
echo -e "${BLUE}=====================================================${NC}"
echo ""

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config FILE    Configuration file (default: configs/dataset_config_v5.yaml)"
    echo "  --splits SPLITS  Splits to generate: train validation test (default: all)"
    echo "  --workers N      Number of parallel workers (default: from config)"
    echo "  --keep-existing  Keep existing files (default: auto-clean before generation)"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Generate complete dataset"
    echo "  $0"
    echo ""
    echo "  # Generate only training split with 16 workers"
    echo "  $0 --splits train --workers 16"
    echo ""
    echo "  # Keep existing files and add to them"
    echo "  $0 --keep-existing"
}

# Default values
CONFIG_FILE="${PROJECT_ROOT}/configs/dataset_config_v5.yaml"
SPLITS=""
WORKERS=""
CLEAN=true  # Auto-clean by default for fresh generation

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --splits)
            shift
            SPLITS="$1"
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --keep-existing)
            CLEAN=false  # Disable auto-clean
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check required packages
echo -e "${YELLOW}Checking dependencies...${NC}"
python3 -c "import yaml, pandas, numpy, soundfile, librosa, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Missing required packages. Installing...${NC}"
    pip install pyyaml pandas numpy soundfile librosa tqdm
fi

# Check data directories
echo -e "${YELLOW}Checking data directories...${NC}"
echo ""

# Check LibriSpeech
if [ -d "/data/LibriSpeech" ]; then
    echo -e "${GREEN}✓ LibriSpeech found${NC}"
    for subset in train-clean-100 train-clean-360 dev-clean dev-other test-clean; do
        if [ -d "/data/LibriSpeech/$subset" ]; then
            count=$(find "/data/LibriSpeech/$subset" -name "*.flac" | wc -l)
            echo -e "  - $subset: $count FLAC files"
        else
            echo -e "  ${YELLOW}⚠ $subset not found${NC}"
        fi
    done
else
    echo -e "${RED}✗ LibriSpeech not found at /data/LibriSpeech${NC}"
    echo "Please download LibriSpeech dataset first"
    exit 1
fi

echo ""

# Check WHAM noise
if [ -d "/data/wham_noise" ]; then
    echo -e "${GREEN}✓ WHAM noise found${NC}"
    for split in tr cv tt; do
        if [ -d "/data/wham_noise/$split" ]; then
            count=$(find "/data/wham_noise/$split" -name "*.wav" | wc -l)
            echo -e "  - $split: $count WAV files"
        else
            echo -e "  ${YELLOW}⚠ $split not found${NC}"
        fi
    done
else
    echo -e "${RED}✗ WHAM noise not found at /data/wham_noise${NC}"
    echo "Please download WHAM noise dataset first"
    exit 1
fi

echo ""

# Auto-clean for fresh generation (unless --keep-existing was used)
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Auto-cleaning previous data for fresh generation...${NC}"
    
    # Clean audio files if they exist
    if [ -d "/data/tse_audio_dataset_v5" ]; then
        existing=$(find /data/tse_audio_dataset_v5 -name "*.wav" 2>/dev/null | wc -l)
        if [ $existing -gt 0 ]; then
            echo "  Removing $existing existing audio files..."
            rm -rf /data/tse_audio_dataset_v5/train/*/*.wav
            rm -rf /data/tse_audio_dataset_v5/validation/*/*.wav
            rm -rf /data/tse_audio_dataset_v5/test/*/*.wav
            rm -f /data/tse_audio_dataset_v5/*/failed_*.json
            echo -e "  ${GREEN}✓ Audio files cleaned${NC}"
        fi
    fi
    
    # Clean metadata for new random combinations
    if [ -d "${PROJECT_ROOT}/metadata_v5" ]; then
        rm -f ${PROJECT_ROOT}/metadata_v5/*.csv
        echo -e "  ${GREEN}✓ Metadata cleaned (will generate new combinations)${NC}"
    fi
    
    # Clean cache
    if [ -d "${PROJECT_ROOT}/cache_v5" ]; then
        rm -f ${PROJECT_ROOT}/cache_v5/*.json
        echo -e "  ${GREEN}✓ Cache cleaned${NC}"
    fi
    
    echo -e "${GREEN}Ready for fresh generation with new random combinations${NC}"
    echo ""
else
    echo -e "${YELLOW}Keeping existing files (--keep-existing flag used)${NC}"
    echo "  Existing metadata will be reused (same speaker combinations)"
    echo "  Existing audio files will be overwritten"
    echo ""
fi

# Build command
CMD="python3 ${PROJECT_ROOT}/src/data/generate_tse_dataset_v5.py"
CMD="$CMD --config $CONFIG_FILE"

if [ ! -z "$SPLITS" ]; then
    CMD="$CMD --splits $SPLITS"
fi

if [ ! -z "$WORKERS" ]; then
    CMD="$CMD --workers $WORKERS"
fi

# Show configuration
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Config file: $CONFIG_FILE"
echo -e "  Splits: ${SPLITS:-all}"
echo -e "  Workers: ${WORKERS:-auto}"
echo ""

# Start generation
echo -e "${GREEN}Starting dataset generation...${NC}"
echo -e "${YELLOW}Command: $CMD${NC}"
echo ""

# Run generation
$CMD

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=====================================================${NC}"
    echo -e "${GREEN}        Dataset Generation Complete!${NC}"
    echo -e "${GREEN}=====================================================${NC}"
    echo ""
    
    # Show statistics
    echo -e "${BLUE}Dataset Statistics:${NC}"
    for split in train validation test; do
        if [ -d "/data/tse_audio_dataset_v5/$split/mixture" ]; then
            count=$(find /data/tse_audio_dataset_v5/$split/mixture -name "*.wav" 2>/dev/null | wc -l)
            if [ $count -gt 0 ]; then
                echo -e "  ${split}: ${GREEN}$count${NC} mixtures"
            fi
        fi
    done
    
    # Check for failures
    echo ""
    failed_files=$(find /data/tse_audio_dataset_v5 -name "failed_*.json" 2>/dev/null)
    if [ ! -z "$failed_files" ]; then
        echo -e "${YELLOW}Note: Some failures detected. Check:${NC}"
        echo "$failed_files"
    else
        echo -e "${GREEN}✅ All samples generated successfully!${NC}"
    fi
    
    # Show output location
    echo ""
    echo -e "${BLUE}Output location:${NC} /data/tse_audio_dataset_v5"
    echo -e "${BLUE}Metadata location:${NC} ${PROJECT_ROOT}/metadata_v5"
    
else
    echo ""
    echo -e "${RED}Dataset generation failed!${NC}"
    echo "Check the logs above for errors"
    exit 1
fi