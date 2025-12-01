# FLOWR.ROOT Data Engineering Pipeline

This document describes the data engineering process for converting filtered docking results into the LMDB format required by FLOWR.ROOT for fine-tuning.

## Overview

The data engineering pipeline transforms high-scoring protein-ligand docking poses into training data for the FLOWR.ROOT generative model. This process involves:

1. **Export**: Converting filtered rescoring results to standardized directory structure
2. **Featurization**: Extracting SE(3)-equivariant features from structures
3. **Serialization**: Writing features to LMDB for high-performance I/O
4. **Statistics**: Computing marginal distributions for prior definition

## Directory Structure

### Input: Rescoring Results
```
rescoring_output/
├── cluster_0001/
│   ├── ligand_001_rescored.json
│   ├── ligand_001_out.sdf
│   └── ...
├── cluster_0002/
│   └── ...
```

### Intermediate: Raw Export
```
flowr_training_data/raw/
├── system_0001/
│   ├── system_0001_protein.pdb
│   └── system_0001_ligand.sdf
├── system_0002/
│   ├── system_0002_protein.pdb
│   └── system_0002_ligand.sdf
├── ...
└── manifest.json
```

### Output: LMDB Database
```
flowr_training_data/final/
├── custom_data.lmdb      # Main database
└── data_statistics.npz   # Prior distribution stats
```

## Quick Start

### From Rescoring Results

```bash
python prepare_flowr_data.py \
    --rescoring_dir rescoring_output \
    --cluster_dir bioemu_clusters \
    --output_dir flowr_training_data \
    --cnn_score_threshold 0.9 \
    --cnn_affinity_threshold -7.0 \
    --parallel
```

### From Existing Raw Directory (Resume)

```bash
python prepare_flowr_data.py \
    --raw_dir flowr_training_data/raw \
    --output_dir flowr_training_data \
    --parallel
```

### Validation

```bash
python prepare_flowr_data.py \
    --validate flowr_training_data/final/custom_data.lmdb
```

## Detailed Pipeline

### Step 1: Pose Export

The `PoseExporter` class converts filtered docking results to the standardized format:

```python
from flowr_data import PoseExporter, ExportConfig

config = ExportConfig(
    output_dir='flowr_training_data/raw',
    copy_receptor=True,
    copy_ligand=True
)
exporter = PoseExporter(config)

# From rescoring results
exported = exporter.export_from_rescoring_results(
    results,        # List of RescoringResult
    cluster_pdb_map # Dict[cluster_name, pdb_path]
)

# Save manifest for resumability
exporter.save_manifest(exported)
```

Each exported system contains:
- `system_XXXX_protein.pdb`: Protein structure (pocket around ligand)
- `system_XXXX_ligand.sdf`: Docked ligand pose

### Step 2: Featurization

The `SystemFeaturizer` extracts features for SE(3)-equivariant networks:

**Protein Features:**
| Feature | Shape | Description |
|---------|-------|-------------|
| `protein_pos` | (N, 3) | Atom positions in Angstroms |
| `protein_elements` | (N,) | Element type indices |
| `protein_amino_acids` | (N,) | Amino acid type indices |
| `protein_edge_index` | (2, E) | KNN graph edges |

**Ligand Features:**
| Feature | Shape | Description |
|---------|-------|-------------|
| `ligand_pos` | (M, 3) | Atom positions in Angstroms |
| `ligand_elements` | (M,) | Element type indices |
| `ligand_hybridization` | (M,) | Hybridization state indices |
| `ligand_formal_charge` | (M,) | Formal charges |
| `ligand_edge_index` | (2, F) | KNN graph edges |

**Additional:**
| Feature | Type | Description |
|---------|------|-------------|
| `affinity` | float32 | Binding affinity (CNN score) |
| `system_id` | string | Unique identifier |

```python
from flowr_data import SystemFeaturizer, FeaturizationConfig

config = FeaturizationConfig(
    knn_k=16,           # K for KNN graph
    pocket_radius=10.0, # Å around ligand
    use_hydrogens=False,
    center_on_ligand=True
)
featurizer = SystemFeaturizer(config)

features = featurizer.featurize_system(
    system_id='system_0001',
    protein_path='system_0001_protein.pdb',
    ligand_path='system_0001_ligand.sdf',
    affinity=-8.5
)
```

### Step 3: LMDB Writing

The `LMDBWriter` serializes features for high-performance I/O:

```python
from flowr_data import LMDBWriter

writer = LMDBWriter(
    output_dir='flowr_training_data/final',
    db_name='custom_data'
)

for features in feature_iterator:
    writer.add_system(features)

stats = writer.finalize()  # Also saves data_statistics.npz
```

### Step 4: Data Statistics

The `data_statistics.npz` file contains marginal distributions:

| Key | Description |
|-----|-------------|
| `n_systems` | Total number of systems |
| `protein_element_probs` | Element type distribution (protein) |
| `ligand_element_probs` | Element type distribution (ligand) |
| `amino_acid_probs` | Amino acid distribution |
| `hybridization_probs` | Hybridization state distribution |
| `affinity_mean/std/min/max` | Affinity statistics |
| `ligand_atom_mean/std` | Ligand size statistics |
| `position_mean/std` | Coordinate statistics |

These statistics define the prior distribution for flow matching.

## SLURM Workflow

For HPC environments, use the provided SLURM scripts:

### 1. Preprocessing (preprocess.sl)

```bash
# Configure paths
export WORKSPACE_DIR=/path/to/CSCI653_Project
export RAW_DIR=${WORKSPACE_DIR}/flowr_training_data/raw
export N_WORKERS=8

# Submit job
sbatch slurm/preprocess.sl
```

### 2. Merging (merge.sl)

If you have multiple batches to combine:

```bash
export LMDB_PATHS="batch1/custom_data.lmdb,batch2/custom_data.lmdb"
export OUTPUT_DIR=flowr_training_data/merged

sbatch slurm/merge.sl
```

### 3. Fine-Tuning (finetune.sl)

```bash
export FLOWR_DIR=/path/to/flowr
export DATA_PATH=${WORKSPACE_DIR}/flowr_training_data/final/custom_data.lmdb
export USE_LORA=True
export LORA_RANK=16

sbatch slurm/finetune.sl
```

## Configuration Reference

### FlowrDataConfig

```python
@dataclass
class FlowrDataConfig:
    output_dir: Path           # Base output directory
    knn_k: int = 16            # K for KNN graph
    pocket_radius: float = 10.0  # Pocket extraction radius (Å)
    use_hydrogens: bool = False  # Include H atoms
    center_on_ligand: bool = True  # Center coordinates
    n_workers: int = 4         # Parallel workers
```

### FeaturizationConfig

```python
@dataclass
class FeaturizationConfig:
    knn_k: int = 16
    max_protein_atoms: int = 2000
    max_ligand_atoms: int = 100
    pocket_radius: float = 10.0
    use_hydrogens: bool = False
    center_on_ligand: bool = True
```

## Vocabulary Definitions

### Protein Elements
```python
PROTEIN_ELEMENTS = ['C', 'N', 'O', 'S', 'H', 'P', 'SE']
```

### Ligand Elements
```python
LIGAND_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'H']
```

### Amino Acids
```python
AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK'
]
```

### Hybridization States
```python
HYBRIDIZATION_STATES = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
```

## API Reference

### Main Functions

```python
# High-level preparation
from flowr_data import FlowrDataPreparer, FlowrDataConfig

config = FlowrDataConfig(output_dir='training_data')
preparer = FlowrDataPreparer(config)

# Complete pipeline
stats = preparer.prepare_from_rescoring_results(results, cluster_map)

# Or step by step
preparer.export_from_rescoring_results(results, cluster_map)
stats = preparer.generate_lmdb(parallel=True)
preparer.validate_lmdb()
```

### LMDB Reading

```python
from flowr_data.lmdb_writer import LMDBReader

reader = LMDBReader('custom_data.lmdb')
print(f"Database contains {len(reader)} systems")

# Random access
features = reader[42]

# Iteration
for features in reader:
    print(features.system_id, features.affinity)

reader.close()
```

## Troubleshooting

### Common Issues

1. **Memory Error during featurization**
   - Reduce `n_workers` for parallel processing
   - Process in smaller batches

2. **Missing atoms in pocket**
   - Increase `pocket_radius` (default 10Å)
   - Check ligand positioning

3. **LMDB write failures**
   - Ensure sufficient disk space
   - Check file permissions

4. **RDKit import errors**
   - Install: `pip install rdkit`
   - For conda: `conda install -c conda-forge rdkit`

### Validation Checklist

- [ ] All systems have non-zero protein atoms
- [ ] All systems have non-zero ligand atoms  
- [ ] Affinities are in expected range (e.g., -15 to 0)
- [ ] Statistics file exists and is readable
- [ ] Random samples can be loaded without errors

## References

- FLOWR.ROOT paper and repository
- LMDB documentation: https://lmdb.readthedocs.io/
- RDKit documentation: https://www.rdkit.org/docs/
- BioPython PDB parser: https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ
