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
FLOWR_DIR="${FLOWR_DIR:-/path/to/flowr}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/path/to/CSCI653_Project}"
DATA_PATH="${DATA_PATH:-${WORKSPACE_DIR}/flowr_training_data/final/custom_data.lmdb}"
CHECKPOINT="${CHECKPOINT:-${FLOWR_DIR}/checkpoints/flowr_root_base.pt}"
CONFIG="${CONFIG:-${WORKSPACE_DIR}/configs/finetune_bioemu.yaml}"
LOG_DIR="${LOG_DIR:-${WORKSPACE_DIR}/logs/finetune_$(date +%Y%m%d_%H%M%S)}"

# LoRA Configuration
USE_LORA="${USE_LORA:-True}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"

# Training Hyperparameters
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"

# Loss Weights
LOSS_WEIGHT_FLOW="${LOSS_WEIGHT_FLOW:-1.0}"
LOSS_WEIGHT_AFFINITY="${LOSS_WEIGHT_AFFINITY:-0.5}"
LOSS_WEIGHT_AUX="${LOSS_WEIGHT_AUX:-0.1}"

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
echo "  Config: ${CONFIG}"
echo "  Log directory: ${LOG_DIR}"
echo ""
echo "LoRA Settings:"
echo "  Use LoRA: ${USE_LORA}"
echo "  Rank: ${LORA_RANK}"
echo "  Alpha: ${LORA_ALPHA}"
echo "  Dropout: ${LORA_DROPOUT}"
echo ""
echo "Training Parameters:"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Warmup steps: ${WARMUP_STEPS}"
echo ""
echo "Loss Weights:"
echo "  Flow: ${LOSS_WEIGHT_FLOW}"
echo "  Affinity: ${LOSS_WEIGHT_AFFINITY}"
echo "  Auxiliary: ${LOSS_WEIGHT_AUX}"
echo "=============================================="

# Load modules (adjust for your HPC system)
# module load python/3.10
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment
source ~/.bashrc
conda activate flowr_env  # Your FLOWR environment

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${LOG_DIR}/checkpoints"
mkdir -p logs

# Verify prerequisites
if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: Data file not found: ${DATA_PATH}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "WARNING: Checkpoint not found: ${CHECKPOINT}"
    echo "Training from scratch..."
    CHECKPOINT=""
fi

# Check GPU
nvidia-smi
echo ""

# Change to FLOWR directory
cd "${FLOWR_DIR}"

# Build training command
TRAIN_CMD="python scripts/train.py"

# Add config if exists
if [ -f "${CONFIG}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --config ${CONFIG}"
fi

# Add checkpoint if exists
if [ -n "${CHECKPOINT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume_from_checkpoint ${CHECKPOINT}"
fi

# Add data path
TRAIN_CMD="${TRAIN_CMD} --data_path ${DATA_PATH}"

# Add LoRA settings
TRAIN_CMD="${TRAIN_CMD} --use_lora ${USE_LORA}"
TRAIN_CMD="${TRAIN_CMD} --lora_rank ${LORA_RANK}"
TRAIN_CMD="${TRAIN_CMD} --lora_alpha ${LORA_ALPHA}"
TRAIN_CMD="${TRAIN_CMD} --lora_dropout ${LORA_DROPOUT}"

# Add training hyperparameters
TRAIN_CMD="${TRAIN_CMD} --learning_rate ${LEARNING_RATE}"
TRAIN_CMD="${TRAIN_CMD} --batch_size ${BATCH_SIZE}"
TRAIN_CMD="${TRAIN_CMD} --num_epochs ${NUM_EPOCHS}"
TRAIN_CMD="${TRAIN_CMD} --warmup_steps ${WARMUP_STEPS}"

# Add loss weights
TRAIN_CMD="${TRAIN_CMD} --loss_weight_flow ${LOSS_WEIGHT_FLOW}"
TRAIN_CMD="${TRAIN_CMD} --loss_weight_affinity ${LOSS_WEIGHT_AFFINITY}"
TRAIN_CMD="${TRAIN_CMD} --loss_weight_aux ${LOSS_WEIGHT_AUX}"

# Add logging
TRAIN_CMD="${TRAIN_CMD} --log_dir ${LOG_DIR}"

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
    echo ""
    echo "Checkpoints:"
    ls -lh "${LOG_DIR}/checkpoints/" 2>/dev/null || echo "No checkpoints saved"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate the model on a validation set"
    echo "  2. Run inference to generate new ligands"
    echo "  3. Validate generated molecules with PoseBusters"
else
    echo ""
    echo "ERROR: Training failed with exit code ${EXIT_STATUS}"
    exit ${EXIT_STATUS}
fi

echo ""
echo "End time: $(date)"
