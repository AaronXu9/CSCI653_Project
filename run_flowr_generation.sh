#!/bin/bash

# Set up paths
PROJECT_ROOT="/home/aoxu/projects/CSCI653_Project"
FLOWR_ROOT="$PROJECT_ROOT/flowr_root"
DATA_DIR="$PROJECT_ROOT/data/5s9l__1__1.A__1.H_1.I"
CKPT_PATH="$FLOWR_ROOT/checkpoints/flowr_root.ckpt"
SAVE_DIR="$PROJECT_ROOT/flowr_output/generation_results"

# Ensure output directory exists
mkdir -p "$SAVE_DIR"

# Activate environment (uncomment if needed, or ensure you are in 'flowr_root' env)
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate flowr_root

# Set PYTHONPATH
export PYTHONPATH="$FLOWR_ROOT:$PYTHONPATH"

# Run generation
# This will generate molecules and automatically predict their affinity
echo "Starting generation and affinity prediction..."
python -m flowr.gen.generate_from_pdb \
    --pdb_file "$DATA_DIR/5s9l__1__1.A__1.H_1.I_protein.pdb" \
    --ligand_file "$DATA_DIR/5s9l__1__1.A__1.H_1.I_ligand.sdf" \
    --arch pocket \
    --pocket_type holo \
    --cut_pocket \
    --pocket_cutoff 7 \
    --gpus 1 \
    --batch_cost 20 \
    --ckpt_path "$CKPT_PATH" \
    --save_dir "$SAVE_DIR" \
    --max_sample_iter 20 \
    --coord_noise_scale 0.1 \
    --sample_n_molecules_per_target 10 \
    --categorical_strategy uniform-sample \
    --filter_valid_unique

echo "Done! Results saved to $SAVE_DIR"
