#!/bin/bash
#SBATCH --job-name=ensemble_docking
#SBATCH --output=logs/docking_%j.out
#SBATCH --error=logs/docking_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# =============================================================================
# Ensemble Docking Pipeline for BioEmu Clusters
# =============================================================================
# This script runs high-throughput docking of a ligand library against
# BioEmu-generated protein conformational ensembles using Uni-Dock.
#
# Prerequisites:
#   - Uni-Dock installed and in PATH
#   - Python environment with required packages (mdtraj, rdkit, numpy)
#   - BioEmu cluster representatives in bioemu_clusters/
#   - Ligand library prepared (SDF format)
#
# Usage:
#   sbatch run_ensemble_docking.sbatch
#
# Or with custom parameters:
#   sbatch --export=LIGAND_LIBRARY=/path/to/ligands.sdf run_ensemble_docking.sbatch
# =============================================================================

set -e

# Configuration - modify these or pass via --export
CLUSTER_DIR="${CLUSTER_DIR:-bioemu_clusters}"
LIGAND_LIBRARY="${LIGAND_LIBRARY:-data/ligand_library.sdf}"
OUTPUT_DIR="${OUTPUT_DIR:-docking_output}"
RESCORE_OUTPUT="${RESCORE_OUTPUT:-rescoring_output}"

# Docking box parameters - MUST be set for your target
# These coordinates should define the binding site
CENTER_X="${CENTER_X:-10.5}"
CENTER_Y="${CENTER_Y:-22.1}"
CENTER_Z="${CENTER_Z:--5.4}"
SIZE_X="${SIZE_X:-20}"
SIZE_Y="${SIZE_Y:-20}"
SIZE_Z="${SIZE_Z:-20}"

# Docking parameters
EXHAUSTIVENESS="${EXHAUSTIVENESS:-128}"
NUM_MODES="${NUM_MODES:-10}"
SCORING="${SCORING:-vina}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-128}"

# Rescoring parameters
CNN_SCORE_THRESHOLD="${CNN_SCORE_THRESHOLD:-0.9}"
CNN_AFFINITY_THRESHOLD="${CNN_AFFINITY_THRESHOLD:--7.0}"

# Create log directory
mkdir -p logs

echo "=============================================="
echo "Ensemble Docking Pipeline"
echo "=============================================="
echo "Start time: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo ""
echo "Configuration:"
echo "  Cluster directory: $CLUSTER_DIR"
echo "  Ligand library: $LIGAND_LIBRARY"
echo "  Output directory: $OUTPUT_DIR"
echo "  Box center: ($CENTER_X, $CENTER_Y, $CENTER_Z)"
echo "  Box size: ($SIZE_X, $SIZE_Y, $SIZE_Z)"
echo "  Exhaustiveness: $EXHAUSTIVENESS"
echo "  GPU batch size: $GPU_BATCH_SIZE"
echo ""

# Activate conda environment if available
if command -v conda &> /dev/null; then
    # Try to activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bioemu 2>/dev/null || conda activate base
fi

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v unidock &> /dev/null; then
    echo "ERROR: Uni-Dock not found in PATH"
    echo "Please install Uni-Dock: https://github.com/dptech-corp/Uni-Dock"
    exit 1
fi
echo "  ✓ Uni-Dock found"

if [ ! -d "$CLUSTER_DIR" ]; then
    echo "ERROR: Cluster directory not found: $CLUSTER_DIR"
    exit 1
fi
N_CLUSTERS=$(ls -1 "$CLUSTER_DIR"/cluster_*.pdb 2>/dev/null | wc -l)
echo "  ✓ Found $N_CLUSTERS cluster representatives"

if [ ! -f "$LIGAND_LIBRARY" ]; then
    echo "ERROR: Ligand library not found: $LIGAND_LIBRARY"
    exit 1
fi
echo "  ✓ Ligand library found"

echo ""
echo "=============================================="
echo "Phase 1: Docking with Uni-Dock"
echo "=============================================="

python run_docking.py \
    --cluster_dir "$CLUSTER_DIR" \
    --ligand_library "$LIGAND_LIBRARY" \
    --output_dir "$OUTPUT_DIR" \
    --center "$CENTER_X" "$CENTER_Y" "$CENTER_Z" \
    --size "$SIZE_X" "$SIZE_Y" "$SIZE_Z" \
    --exhaustiveness "$EXHAUSTIVENESS" \
    --num_modes "$NUM_MODES" \
    --scoring "$SCORING" \
    --gpu_batch_size "$GPU_BATCH_SIZE"

DOCKING_EXIT=$?

if [ $DOCKING_EXIT -ne 0 ]; then
    echo "ERROR: Docking failed with exit code $DOCKING_EXIT"
    exit $DOCKING_EXIT
fi

echo ""
echo "=============================================="
echo "Phase 2: Rescoring with Gnina"
echo "=============================================="

# Check if Gnina is available
if command -v gnina &> /dev/null; then
    echo "Gnina found, proceeding with CNN rescoring..."
    
    python run_rescoring.py \
        --docking_dir "$OUTPUT_DIR" \
        --cluster_dir "$CLUSTER_DIR" \
        --output_dir "$RESCORE_OUTPUT" \
        --cnn_scoring rescore \
        --cnn_score_threshold "$CNN_SCORE_THRESHOLD" \
        --cnn_affinity_threshold "$CNN_AFFINITY_THRESHOLD" \
        --export_csv \
        --export_json \
        --export_sdf \
        --top_n_per_ligand 1

    RESCORE_EXIT=$?
    
    if [ $RESCORE_EXIT -ne 0 ]; then
        echo "WARNING: Rescoring failed with exit code $RESCORE_EXIT"
        echo "Docking results are still available in $OUTPUT_DIR"
    fi
else
    echo "WARNING: Gnina not found, skipping CNN rescoring"
    echo "Install Gnina for improved pose quality filtering"
    echo "https://github.com/gnina/gnina"
fi

echo ""
echo "=============================================="
echo "Pipeline Complete"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  Docking output: $OUTPUT_DIR/"
echo "  Rescoring output: $RESCORE_OUTPUT/"
echo ""
echo "Next steps:"
echo "  1. Review rescoring_stats.json for summary"
echo "  2. Use filtered_poses.sdf for training data"
echo "  3. Optionally apply additional filters"
