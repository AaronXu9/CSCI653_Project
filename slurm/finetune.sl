#!/bin/bash
#SBATCH --job-name=flowr_finetune
#SBATCH --output=logs/flowr_finetune_%j.out
#SBATCH --error=logs/flowr_finetune_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

###############################################################################
# FLOWR.ROOT Fine-Tuning Script
#
# This script fine-tunes the FLOWR.ROOT foundation model on custom
# protein-ligand data using LoRA (Low-Rank Adaptation).
#
# Fine-tuning Strategy:
#   - LoRA adapters on cross-attention layers (ligand → protein)
#   - Frozen backbone to preserve chemical validity rules
#   - Unfrozen affinity head for target-specific calibration
#   - Combined loss: flow matching + affinity prediction
#
# Prerequisites:
#   - FLOWR.ROOT installation with training scripts
#   - Preprocessed LMDB data (custom_data.lmdb)
#   - Pre-trained checkpoint (flowr_root_base.pt)
#
# Usage:
#   sbatch finetune.sl
#
# Or with custom config:
#   CONFIG=configs/my_config.yaml sbatch finetune.sl
###############################################################################

# Configuration
FLOWR_DIR="${FLOWR_DIR:-$(pwd)/flowr_root}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
DATA_PATH="${DATA_PATH:-${WORKSPACE_DIR}/flowr_training_data/final/custom_data.lmdb}"
CHECKPOINT="${CHECKPOINT:-${FLOWR_DIR}/checkpoints/flowr_root.ckpt}"
LOG_DIR="${LOG_DIR:-${WORKSPACE_DIR}/logs/finetune_$(date +%Y%m%d_%H%M%S)}"

# LoRA Configuration
USE_LORA="${USE_LORA:-True}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# Training Hyperparameters
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_COST="${BATCH_COST:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"

# Loss Weights
LOSS_WEIGHT_COORD="${LOSS_WEIGHT_COORD:-1.0}"
LOSS_WEIGHT_TYPE="${LOSS_WEIGHT_TYPE:-1.0}"
LOSS_WEIGHT_BOND="${LOSS_WEIGHT_BOND:-2.0}"
LOSS_WEIGHT_AFFINITY="${LOSS_WEIGHT_AFFINITY:-1.0}"

# Environment setup
echo "=============================================="
echo "FLOWR.ROOT Fine-Tuning"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  FLOWR directory: ${FLOWR_DIR}"
echo "  Data path: ${DATA_PATH}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Log directory: ${LOG_DIR}"
echo ""
echo "LoRA Settings:"
echo "  Use LoRA: ${USE_LORA}"
echo "  Rank: ${LORA_RANK}"
echo "  Alpha: ${LORA_ALPHA}"
echo ""
echo "Training Parameters:"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch cost: ${BATCH_COST}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Warmup steps: ${WARMUP_STEPS}"
echo ""
echo "Loss Weights:"
echo "  Coord: ${LOSS_WEIGHT_COORD}"
echo "  Type: ${LOSS_WEIGHT_TYPE}"
echo "  Bond: ${LOSS_WEIGHT_BOND}"
echo "  Affinity: ${LOSS_WEIGHT_AFFINITY}"
echo "=============================================="

# Activate conda environment
source ~/.bashrc
# Use mamba activate if available, otherwise conda
if command -v mamba &> /dev/null; then
    mamba activate flowr_root
else
    conda activate flowr_root
fi

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p logs

# Verify prerequisites
if [ ! -e "${DATA_PATH}" ]; then
    echo "ERROR: Data path not found: ${DATA_PATH}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "WARNING: Checkpoint not found: ${CHECKPOINT}"
    echo "Training from scratch (or using random init if no ckpt provided)..."
    CHECKPOINT_ARG=""
else
    CHECKPOINT_ARG="--ckpt_path ${CHECKPOINT}"
fi

# Change to FLOWR directory
cd "${FLOWR_DIR}"

# Build training command
TRAIN_CMD="python -m flowr.finetune"

# Add data path
TRAIN_CMD="${TRAIN_CMD} --data_path ${DATA_PATH}"
TRAIN_CMD="${TRAIN_CMD} --dataset custom"

# Add checkpoint if exists
if [ -n "${CHECKPOINT_ARG}" ]; then
    TRAIN_CMD="${TRAIN_CMD} ${CHECKPOINT_ARG}"
fi

# Add LoRA settings
if [ "${USE_LORA}" = "True" ]; then
    TRAIN_CMD="${TRAIN_CMD} --lora_finetuning"
    TRAIN_CMD="${TRAIN_CMD} --lora_rank ${LORA_RANK}"
    TRAIN_CMD="${TRAIN_CMD} --lora_alpha ${LORA_ALPHA}"
fi

# Add training hyperparameters
TRAIN_CMD="${TRAIN_CMD} --lr ${LEARNING_RATE}"
TRAIN_CMD="${TRAIN_CMD} --batch_cost ${BATCH_COST}"
TRAIN_CMD="${TRAIN_CMD} --epochs ${NUM_EPOCHS}"
TRAIN_CMD="${TRAIN_CMD} --warm_up_steps ${WARMUP_STEPS}"
TRAIN_CMD="${TRAIN_CMD} --save_dir ${LOG_DIR}"

# Add required arguments
TRAIN_CMD="${TRAIN_CMD} --pocket_noise fix"
TRAIN_CMD="${TRAIN_CMD} --pocket_coord_noise_std 0.0"

# Add loss weights
TRAIN_CMD="${TRAIN_CMD} --coord_loss_weight ${LOSS_WEIGHT_COORD}"
TRAIN_CMD="${TRAIN_CMD} --type_loss_weight ${LOSS_WEIGHT_TYPE}"
TRAIN_CMD="${TRAIN_CMD} --bond_loss_weight ${LOSS_WEIGHT_BOND}"

# Add affinity finetuning
# Note: Cannot use both LoRA and affinity_finetuning (mutually exclusive in scriptutil.py)
# We prioritize LoRA here. Affinity loss is still calculated via --affinity_loss_weight.
if [ "${USE_LORA}" = "False" ]; then
    TRAIN_CMD="${TRAIN_CMD} --affinity_finetuning pkd"
fi
TRAIN_CMD="${TRAIN_CMD} --affinity_loss_weight ${LOSS_WEIGHT_AFFINITY}"

# Print and execute command
echo "Training command:"
echo "${TRAIN_CMD}"
echo ""
echo "=============================================="
echo "Starting training..."
echo "=============================================="

${TRAIN_CMD} 2>&1 | tee "${LOG_DIR}/training.log"

# Check exit status
EXIT_STATUS=$?

if [ ${EXIT_STATUS} -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Training completed successfully!"
    echo "=============================================="
    echo ""
    echo "Output files:"
    ls -lh "${LOG_DIR}/"
else
    echo ""
    echo "ERROR: Training failed with exit code ${EXIT_STATUS}"
    exit ${EXIT_STATUS}
fi

echo ""
echo "End time: $(date)"
