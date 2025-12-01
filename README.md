
# Dynamic Generative Structure-Based Drug Design Pipeline

## Overview
<!-- [Diagram for the Dynamic Generative Structure-Based Drug Design Pipeline](figs/diagram.svg) -->

Traditional SBDD relies on static crystal structures, which often represent single, low-energy minima and miss transient, bioactive conformations such as "cryptic" pockets. To overcome this "static trap," this pipeline integrates biomolecular emulation with geometric deep learning

We utilize **BioEmu** to generate thermodynamic ensembles of the target protein directly from sequence, capturing functional motions and rare states. These ensembles are used to create a synthetic training dataset to fine-tune **FLOWR.ROOT**, an SE(3)-equivariant flow matching generative model This results in a generator explicitly tailored to the dynamic conformational landscape of the target protein

<!-- Pipeline Workflow

The following diagram outlines the four-phase process implemented in this repository: -->

<img src="figs/diagram.svg" alt="Dynamic Generative Structure-Based Drug Design Pipeline">

## Requirements & Installation

### Core Software Stack

The following versions are recommended for the pipeline: 

| Component | Software | Version | Role |
| :--- | :--- | :--- | :--- |
| **Dynamics** | BioEmu | v1.1 | Protein Ensemble Generation |
| **Clustering** | MDTraj / PyEMMA | 1.9.9 / 2.5.7 | Ensemble Reduction & Selection |
| **Docking** | Uni-Dock | v1.1 | High-Throughput Pose Generation |
| **Rescoring** | Gnina | v1.0+ | High-Fidelity Affinity Filtering |
| **Generator** | FLOWR.ROOT | v1.0 | Ligand Design & Affinity Prediction |
| **Validation** | PoseBusters | v0.2.0 | Physical & Chemical Sanity Checks |

### Environment Setup

It is recommended to use Conda to manage dependencies. The `bioemu` package requires specific extensions for MD tools used in refinement

```bash
# Create environment
conda create -n bioemu_env python=3.10
conda activate bioemu_env

# Installation of the core package and MD extension
pip install bioemu
pip install bioemu[md]   # Installs HPacker, OpenMM, and other MD tools
```

<!-- This repository houses a state-of-the-art computational pipeline for Structure-Based Drug Design (SBDD) that transitions from traditional static methods toward a dynamic, probabilistic understanding of molecular recognition -->



## Detailed Functionality

### Phase I: Constructing the Target Ensemble with BioEmu
Instead of relying on a single PDB file, we generate a high-quality structural ensemble that reflects the protein's solution-state dynamics.
* **BioEmu Sampling:** We use BioEmu trained on aggregate MD data to predict the equilibrium distribution of structures from the amino acid sequence. We typically generate 5,000+ samples to populate the tails of the distribution where cryptic states reside.
* **Refinement:** Raw samples undergo sidechain repacking and NVT equilibration (using OpenMM) to resolve subtle clashes and ensure physical viability.
* **Ensemble Reduction:** To create a diverse, representative training set, refined structures are clustered based on binding pocket RMSD using MDTraj, reducing the ensemble to approximately 50 distinct representative states.

### Phase II: Generating Synthetic Training Data
FLOWR.ROOT requires protein-ligand pairs for supervised training. We generate a high-fidelity "Synthetic Holo-Set" by docking diverse libraries into our ensemble representatives
* **Massive Batch Docking:** We utilize **Uni-Dock** for its extreme GPU-accelerated throughput to execute large-scale virtual screening against all cluster representatives.
* **High-Fidelity Rescoring:** Generated poses are rescored using **Gnina's** deep learning CNN scoring function, which is superior at distinguishing real biological binding modes from artifacts.
* **Data Engineering:** Poses are filtered for high CNN scores (>0.9) and predicted affinity, then featurized and serialized into LMDB format for high-performance I/O during training

```bash
# Complete pipeline
python run_pipeline.py \
    --cluster_dir bioemu_clusters \
    --ligand_library your_ligands.sdf \
    --output_dir output \
    --center 10.5 22.1 -5.4

# Or use SLURM
sbatch run_ensemble_docking.sbatch
```

#### Data Engineering for FLOWR.ROOT

After rescoring, convert filtered poses to FLOWR.ROOT training format:

```bash
# Prepare LMDB database from rescoring results
python prepare_flowr_data.py \
    --rescoring_dir rescoring_output \
    --cluster_dir bioemu_clusters \
    --output_dir flowr_training_data \
    --cnn_score_threshold 0.9 \
    --cnn_affinity_threshold -7.0 \
    --parallel

# Or use SLURM for large datasets
sbatch slurm/preprocess.sl
```

**Output Structure:**
```
flowr_training_data/
├── raw/                    # Exported pose files
│   ├── system_0001/
│   │   ├── system_0001_protein.pdb
│   │   └── system_0001_ligand.sdf
│   └── manifest.json
└── final/
    ├── custom_data.lmdb    # Training database
    └── data_statistics.npz # Prior distribution stats
```

See [docs/FLOWR_DATA_ENGINEERING.md](docs/FLOWR_DATA_ENGINEERING.md) for detailed documentation.

### Phase III: Fine-Tuning FLOWR.ROOT
We adapt the generalist FLOWR.ROOT foundation model to the specific geometric and electrostatic boundary conditions of the target protein's dynamic pocket
* **Flow Matching Architecture:** FLOWR.ROOT uses Continuous Normalizing Flows to learn a time-dependent vector field that transports a prior noise distribution to valid ligand structures, respecting SE(3)-equivariance
* **Low-Rank Adaptation (LoRA):** To prevent catastrophic forgetting of general chemical rules, we apply LoRA adapters to the model's cross-attention layers while freezing the backbone weights
* **Joint Training:** The model is trained with a combined loss function, optimizing both the flow matching objective for structure generation and MSE loss for the joint affinity prediction head

```bash
# Fine-tune FLOWR.ROOT with LoRA
python scripts/train.py \
    --config configs/finetune_bioemu.yaml \
    --resume_from_checkpoint checkpoints/flowr_root_base.pt \
    --data_path flowr_training_data/final/custom_data.lmdb \
    --use_lora True \
    --lora_rank 16 \
    --loss_weight_flow 1.0 \
    --loss_weight_affinity 0.5 \
    --log_dir logs/finetune_bioemu_v1

# Or use SLURM (A100 GPU recommended)
sbatch slurm/finetune.sl
```

**Fine-Tuning Configuration:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-5 | Low rate to prevent forgetting |
| Batch Size | 16-32 | Memory-intensive SE(3) models |
| LoRA Rank | 16 | Balance adaptability vs. parameters |
| Target Layers | Cross-attention | Learn new protein-ligand rules |
| Epochs | 50-100 | Monitor validation loss |

### Phase IV: Inference, Steering, and Validation
Once fine-tuned, the model serves as a bespoke generator for the target, followed by rigorous physics-based validation
* **Affinity Steering:** During inference, we generate thousands of trajectories by solving the ODE and use the trained affinity head to perform importance sampling, prioritizing high-affinity "super-binders"
* **PoseBusters Validation:** Generated ligands are subjected to the **PoseBusters** suite to ensure chemical validity, planarity, correct bond geometry, and the absence of severe protein-ligand clashes
* **Redocking Consistency:** As a final orthogonal check, valid ligands are redocked using Gnina. A low Self-Consistency RMSD (< 2.0 Å) indicates the generative model found a stable energy minimum supported by physics-based docking
