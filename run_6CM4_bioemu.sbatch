#!/bin/bash
#SBATCH --job-name=bioemu_6CM4
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1        # request one A100 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00               # adjust if needed
#SBATCH --output=logs/bioemu_6CM4_%j.out
#SBATCH --error=logs/bioemu_6CM4_%j.err

# --- setup ---

# Load conda (adjust this to your setup)
source ~/miniconda3/etc/profile.d/conda.sh   # or ~/.conda/etc/profile.d/conda.sh
conda activate bioemu

# Make sure log directory exists
mkdir -p logs
mkdir -p ./output/6CM4/

# Optional: show which GPU we got
echo "Running on node: $(hostname)"
nvidia-smi

# --- run bioemu ---

python -m bioemu.sample \
  --sequence 6CM4.fasta \
  --num_samples 10000 \
  --output_dir ./output/6CM4/
