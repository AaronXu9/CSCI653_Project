#!/bin/bash
#SBATCH --job-name=flowr_preprocess
#SBATCH --output=logs/flowr_preprocess_%j.out
#SBATCH --error=logs/flowr_preprocess_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=main

###############################################################################
# FLOWR.ROOT Data Preprocessing Script
#
# This script performs featurization of protein-ligand pairs for FLOWR.ROOT
# training. It converts the exported raw data into the LMDB format.
#
# Features extracted:
#   Protein: atom positions, element types, amino acid types, KNN edges
#   Ligand: atom positions, element types, hybridization, formal charge, KNN edges
#
# Usage:
#   sbatch preprocess.sl
#
# Prerequisites:
#   - Exported raw data in training_data/raw/
#   - Python environment with required packages
###############################################################################

# Configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-/path/to/CSCI653_Project}"
RAW_DIR="${RAW_DIR:-${WORKSPACE_DIR}/flowr_training_data/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/flowr_training_data}"
POCKET_RADIUS="${POCKET_RADIUS:-10.0}"
KNN_K="${KNN_K:-16}"
N_WORKERS="${N_WORKERS:-8}"

# Environment setup
echo "=============================================="
echo "FLOWR.ROOT Data Preprocessing"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Workspace: ${WORKSPACE_DIR}"
echo "  Raw data: ${RAW_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Pocket radius: ${POCKET_RADIUS} Å"
echo "  KNN K: ${KNN_K}"
echo "  Workers: ${N_WORKERS}"
echo "=============================================="

# Load modules (adjust for your HPC system)
# module load python/3.10
# module load cuda/11.8

# Activate conda environment
source ~/.bashrc
conda activate bioemu_env  # Or your FLOWR environment

# Create log directory
mkdir -p logs

# Change to workspace
cd "${WORKSPACE_DIR}"

# Check if raw data exists
if [ ! -d "${RAW_DIR}" ]; then
    echo "ERROR: Raw data directory not found: ${RAW_DIR}"
    exit 1
fi

if [ ! -f "${RAW_DIR}/manifest.json" ]; then
    echo "ERROR: Manifest file not found: ${RAW_DIR}/manifest.json"
    exit 1
fi

# Count systems
N_SYSTEMS=$(python -c "import json; print(len(json.load(open('${RAW_DIR}/manifest.json'))['systems']))")
echo "Found ${N_SYSTEMS} systems to process"

# Run preprocessing
echo ""
echo "Starting featurization..."
echo ""

python prepare_flowr_data.py \
    --raw_dir "${RAW_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --pocket_radius "${POCKET_RADIUS}" \
    --knn_k "${KNN_K}" \
    --n_workers "${N_WORKERS}" \
    --parallel \
    -v

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Preprocessing completed successfully!"
    echo "=============================================="
    echo ""
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}/final/"
    echo ""
    echo "To validate:"
    echo "  python prepare_flowr_data.py --validate ${OUTPUT_DIR}/final/custom_data.lmdb"
else
    echo ""
    echo "ERROR: Preprocessing failed!"
    exit 1
fi

echo ""
echo "End time: $(date)"
