#!/bin/bash

# Default values
DEFAULT_INPUTS="coralscapes/leftImg8bit/val"
DEFAULT_CONFIG="configs/segformer-mit-b2.yaml"
DEFAULT_CHECKPOINT="model_checkpoints/final_model_checkpoints/segformer_mit_b2_epoch265"

# Help message
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --inputs PATH      Path to input images or directory (default: $DEFAULT_INPUTS)"
    echo "  -c, --config PATH      Path to config yaml (default: $DEFAULT_CONFIG)"
    echo "  -m, --model PATH       Path to model checkpoint (default: $DEFAULT_CHECKPOINT)"
    echo "  -e, --env NAME         Conda environment name (default: coralscapes)"
    echo "  -h, --help             Show this help message"
    exit 1
}

# Parse arguments
INPUTS=$DEFAULT_INPUTS
CONFIG=$DEFAULT_CONFIG
CHECKPOINT=$DEFAULT_CHECKPOINT
ENV_NAME="coralscapes"

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--inputs) INPUTS="$2"; shift 2 ;;
        -c|--config) CONFIG="$2"; shift 2 ;;
        -m|--model) CHECKPOINT="$2"; shift 2 ;;
        -e|--env) ENV_NAME="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Ensure absolute paths for inputs (relative to project root if needed)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Check if paths exist
if [ ! -d "$INPUTS" ] && [ ! -f "$INPUTS" ]; then
    echo "Error: Input path '$INPUTS' not found."
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file '$CONFIG' not found."
    exit 1
fi

if [ ! -f "$CHECKPOINT" ] && [ ! -d "$CHECKPOINT" ]; then
    echo "Error: Checkpoint '$CHECKPOINT' not found."
    exit 1
fi

echo "============================================================"
echo "      CORALSCAPES INFERENCE RUNNER"
echo "============================================================"
echo "Env        : $ENV_NAME"
echo "Inputs     : $INPUTS"
echo "Config     : $CONFIG"
echo "Checkpoint : $CHECKPOINT"
echo "Root       : $ROOT_DIR"
echo "============================================================"

# Run the inference script
# Using conda run --no-capture-output to ensure real-time logging
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd scripts
python -u inference_evaluation.py \
    --inputs "../$INPUTS" \
    --config "../$CONFIG" \
    --model-checkpoint "../$CHECKPOINT"

echo "============================================================"
echo "Finished! Results and logs are in work_dir/"
echo "============================================================"
