#!/bin/bash
#SBATCH --job-name=flowr_merge
#SBATCH --output=logs/flowr_merge_%j.out
#SBATCH --error=logs/flowr_merge_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=main

###############################################################################
# FLOWR.ROOT LMDB Merge Script
#
# This script merges multiple LMDB databases into a single database.
# Use this when you have processed data in multiple batches or from
# different sources that need to be combined for training.
#
# Output:
#   - merged_data.lmdb: Combined database
#   - data_statistics.npz: Updated statistics for combined data
#
# Usage:
#   sbatch merge.sl
#
# Or with custom paths:
#   LMDB_PATHS="db1.lmdb,db2.lmdb" OUTPUT_DIR="merged" sbatch merge.sl
###############################################################################

# Configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-/path/to/CSCI653_Project}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/flowr_training_data/merged}"
DB_NAME="${DB_NAME:-merged_data}"

# Comma-separated list of LMDB paths to merge
# Example: "batch1/custom_data.lmdb,batch2/custom_data.lmdb"
LMDB_PATHS="${LMDB_PATHS:-}"

# Environment setup
echo "=============================================="
echo "FLOWR.ROOT LMDB Merge"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Workspace: ${WORKSPACE_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Database name: ${DB_NAME}"
echo "=============================================="

# Load modules
# module load python/3.10

# Activate conda environment
source ~/.bashrc
conda activate bioemu_env

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Change to workspace
cd "${WORKSPACE_DIR}"

# If LMDB_PATHS not set, find all custom_data.lmdb files
if [ -z "${LMDB_PATHS}" ]; then
    echo "No LMDB_PATHS specified, searching for databases..."
    LMDB_FILES=$(find flowr_training_data -name "custom_data.lmdb" -type f | sort)
    
    if [ -z "${LMDB_FILES}" ]; then
        echo "ERROR: No LMDB databases found!"
        exit 1
    fi
    
    echo "Found databases:"
    echo "${LMDB_FILES}"
else
    # Convert comma-separated to newline-separated
    LMDB_FILES=$(echo "${LMDB_PATHS}" | tr ',' '\n')
    echo "Merging specified databases:"
    echo "${LMDB_FILES}"
fi

# Count databases
N_DBS=$(echo "${LMDB_FILES}" | wc -l)
echo ""
echo "Total databases to merge: ${N_DBS}"

# Create Python merge script
python << 'MERGE_SCRIPT'
import os
import sys
sys.path.insert(0, os.getcwd())

from pathlib import Path
from flowr_data.lmdb_writer import merge_lmdb_databases

# Get paths from environment
lmdb_files = os.environ.get('LMDB_FILES', '').strip().split('\n')
lmdb_files = [Path(f.strip()) for f in lmdb_files if f.strip()]
output_dir = Path(os.environ.get('OUTPUT_DIR', 'flowr_training_data/merged'))
db_name = os.environ.get('DB_NAME', 'merged_data')

print(f"\nMerging {len(lmdb_files)} databases...")
for f in lmdb_files:
    print(f"  - {f}")

# Validate all files exist
for f in lmdb_files:
    if not f.exists():
        print(f"ERROR: Database not found: {f}")
        sys.exit(1)

# Merge databases
stats = merge_lmdb_databases(lmdb_files, output_dir, db_name)

print(f"\nMerge complete!")
print(f"Total systems: {stats.n_systems}")
print(f"Output: {output_dir / f'{db_name}.lmdb'}")

# Print statistics
if stats.affinity_stats:
    print(f"\nAffinity distribution:")
    print(f"  Mean: {stats.affinity_stats.get('mean', 'N/A'):.2f}")
    print(f"  Std:  {stats.affinity_stats.get('std', 'N/A'):.2f}")
    print(f"  Range: [{stats.affinity_stats.get('min', 'N/A'):.2f}, "
          f"{stats.affinity_stats.get('max', 'N/A'):.2f}]")
MERGE_SCRIPT

# Export variables for Python script
export LMDB_FILES="${LMDB_FILES}"
export OUTPUT_DIR="${OUTPUT_DIR}"
export DB_NAME="${DB_NAME}"

# Run the merge script
python << MERGE_SCRIPT_END
import os
import sys
sys.path.insert(0, os.getcwd())

from pathlib import Path
from flowr_data.lmdb_writer import merge_lmdb_databases

# Get paths from environment
lmdb_files_str = os.environ.get('LMDB_FILES', '').strip()
lmdb_files = [Path(f.strip()) for f in lmdb_files_str.split('\n') if f.strip()]
output_dir = Path(os.environ.get('OUTPUT_DIR', 'flowr_training_data/merged'))
db_name = os.environ.get('DB_NAME', 'merged_data')

print(f"\nMerging {len(lmdb_files)} databases...")
for f in lmdb_files:
    print(f"  - {f}")

# Validate all files exist
for f in lmdb_files:
    if not f.exists():
        print(f"ERROR: Database not found: {f}")
        sys.exit(1)

# Merge databases
stats = merge_lmdb_databases(lmdb_files, output_dir, db_name)

print(f"\nMerge complete!")
print(f"Total systems: {stats.n_systems}")
print(f"Output: {output_dir / f'{db_name}.lmdb'}")

# Print statistics
if stats.affinity_stats:
    print(f"\nAffinity distribution:")
    print(f"  Mean: {stats.affinity_stats.get('mean', 'N/A'):.2f}")
    print(f"  Std:  {stats.affinity_stats.get('std', 'N/A'):.2f}")
    print(f"  Range: [{stats.affinity_stats.get('min', 'N/A'):.2f}, "
          f"{stats.affinity_stats.get('max', 'N/A'):.2f}]")
MERGE_SCRIPT_END

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Merge completed successfully!"
    echo "=============================================="
    echo ""
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}/"
else
    echo ""
    echo "ERROR: Merge failed!"
    exit 1
fi

echo ""
echo "End time: $(date)"
