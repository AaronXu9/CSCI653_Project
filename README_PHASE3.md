# Phase 3: Data Preparation and FLOWR.ROOT Fine-Tuning

This document outlines the steps to prepare data and fine-tune the FLOWR.ROOT model, corresponding to Phase 3 of the project.

## 1. Data Preparation

The `prepare_flowr_data.py` script has been enhanced to support:
- Reading scores directly from SDF tags (useful when JSON results are missing).
- Handling both pKd (positive, higher is better) and binding energy (negative, lower is better) for affinity filtering.

### Usage

To prepare the data from your rescoring output:

```bash
# If you have JSON results (standard pipeline output)
python prepare_flowr_data.py \
    --rescoring_dir rescoring_output \
    --cluster_dir bioemu_clusters \
    --output_dir flowr_training_data \
    --cnn_score_threshold 0.9 \
    --cnn_affinity_threshold 6.0  # pKd threshold (e.g. 6.0 = 1uM)

# If you only have SDF files (e.g. from manual Gnina run)
python prepare_flowr_data.py \
    --docking_dir rescoring_output \
    --cluster_dir bioemu_clusters \
    --output_dir flowr_training_data \
    --cnn_score_threshold 0.9 \
    --cnn_affinity_threshold 6.0
```

**Note on Affinity Threshold:**
- If `cnn_affinity_threshold` is **positive** (e.g., 6.0), it is treated as **pKd** (Gnina default), and poses with `affinity < threshold` are filtered out.
- If `cnn_affinity_threshold` is **negative** (e.g., -7.0), it is treated as **binding energy** (kcal/mol), and poses with `affinity > threshold` are filtered out.

### Output

The script generates:
- `flowr_training_data/final/custom_data.lmdb`: The LMDB database for training.
- `flowr_training_data/final/data_statistics.npz`: Statistics for the prior distribution.

## 2. Fine-Tuning FLOWR.ROOT

The fine-tuning process uses the generated LMDB data to adapt the FLOWR.ROOT model.

### Configuration

The configuration is defined in `configs/finetune_bioemu.yaml`. You can adjust hyperparameters like learning rate, batch size, and LoRA rank there.

### Running the Job

Use the provided SLURM script to submit the job:

```bash
# Set the path to your FLOWR installation if not in default location
export FLOWR_DIR=/path/to/flowr

# Submit the job
sbatch slurm/finetune.sl
```

The script will:
1. Load the data from `flowr_training_data/final/custom_data.lmdb`.
2. Load the pre-trained checkpoint.
3. Fine-tune the model using LoRA.
4. Save checkpoints and logs to `logs/`.

## Prerequisites

Ensure you have the necessary dependencies installed. We use the `flowr_root` repository as a submodule.

### 1. Set up the Submodule
If you haven't already, initialize the submodule:
```bash
git submodule update --init --recursive
```

### 2. Create the Environment
We have provided a fixed environment configuration `environment_flowr.yml` that resolves dependency conflicts (specifically moving PyTorch and Lightning to pip installation).

```bash
mamba env create -f environment_flowr.yml
mamba activate flowr_root
```

### 3. Dependencies List
- `rdkit`
- `lmdb`
- `numpy`
- `scipy`
- `pytorch` (via pip)
- `lightning` (via pip)
- `bioemu` (for Phase 1)
- `gnina` (for Phase 2)
- `flowr` (for Phase 3 fine-tuning)
