# Ensemble Docking and Rescoring Pipeline

This documentation covers the Phase II components of the drug discovery pipeline: generating synthetic training data via ensemble docking and CNN rescoring.

## Overview

The pipeline consists of two main stages:
1. **Docking (Uni-Dock)**: High-throughput pose generation against BioEmu cluster representatives
2. **Rescoring (Gnina)**: CNN-based filtering for high-quality training data

## Architecture

The codebase follows an extensible design pattern:

```
docking/
├── __init__.py          # Public API exports
├── base.py              # Abstract base classes (DockingEngine, DockingConfig, DockingResult)
├── unidock.py           # Uni-Dock implementation
└── utils.py             # Utilities (file conversion, box detection)

rescoring/
├── __init__.py          # Public API exports
├── base.py              # Abstract base classes (RescoringEngine, RescoringConfig, etc.)
├── gnina.py             # Gnina CNN rescoring implementation
├── filters.py           # Multi-criteria filtering utilities
└── custom_template.py   # Template for adding new rescoring methods

run_docking.py           # Batch docking CLI
run_rescoring.py         # Batch rescoring CLI
run_pipeline.py          # Complete pipeline orchestration
run_ensemble_docking.sbatch  # SLURM submission script
```

## Quick Start

### 1. Basic Docking

```bash
python run_docking.py \
    --cluster_dir bioemu_clusters \
    --ligand_library ligands.sdf \
    --output_dir docking_output \
    --center 10.5 22.1 -5.4 \
    --size 20 20 20 \
    --exhaustiveness 128
```

### 2. Basic Rescoring

```bash
python run_rescoring.py \
    --docking_dir docking_output \
    --cluster_dir bioemu_clusters \
    --output_dir rescoring_output \
    --cnn_score_threshold 0.9 \
    --export_csv
```

### 3. Complete Pipeline

```bash
python run_pipeline.py \
    --cluster_dir bioemu_clusters \
    --ligand_library ligands.sdf \
    --output_dir pipeline_output \
    --center 10.5 22.1 -5.4
```

### 4. SLURM Submission

```bash
# Edit parameters in the script first
sbatch run_ensemble_docking.sbatch
```

## Docking Module

### UniDockConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `receptor_path` | Path | required | PDB or PDBQT receptor file |
| `ligand_paths` | Path/List | required | Ligand file(s) or index file |
| `output_dir` | Path | required | Output directory |
| `center_x/y/z` | float | required | Docking box center |
| `size_x/y/z` | float | 20.0 | Docking box dimensions |
| `exhaustiveness` | int | 32 | Search exhaustiveness |
| `num_modes` | int | 9 | Poses per ligand |
| `scoring` | str | 'vina' | Scoring function: vina/vinardo/ad4 |
| `gpu_batch_size` | int | 128 | GPU batch size |

### Programmatic Usage

```python
from docking import UniDockEngine, UniDockConfig

config = UniDockConfig(
    receptor_path='receptor.pdb',
    ligand_paths='ligands.sdf',
    output_dir='output',
    center_x=10.0, center_y=20.0, center_z=30.0,
    exhaustiveness=128,
    gpu_batch_size=256
)

engine = UniDockEngine(config)
results = engine.dock()

for result in results:
    print(f"{result.ligand_name}: {result.best_score:.2f} kcal/mol")
```

### Automatic Box Detection

```python
from docking.utils import detect_binding_site, get_box_from_reference_ligand

# From residue selection
center, size = detect_binding_site(
    'receptor.pdb',
    method='residues',
    residue_ids=[99, 100, 101, 145, 146, 147]
)

# From reference ligand
center, size = get_box_from_reference_ligand(
    'reference_ligand.sdf',
    padding=5.0
)
```

## Rescoring Module

### GninaConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `receptor_path` | Path | required | Receptor PDB file |
| `poses_path` | Path | required | Poses to rescore |
| `output_dir` | Path | required | Output directory |
| `cnn_scoring` | str | 'rescore' | CNN mode: none/rescore/refinement/all |
| `cnn_score_threshold` | float | 0.0 | Minimum CNN score filter |
| `cnn_affinity_threshold` | float | None | Maximum affinity filter |
| `minimize` | bool | False | Minimize before scoring |
| `no_gpu` | bool | False | Disable GPU |

### Gnina Scores

- **CNNscore** (0-1): Probability that pose is a true binding pose. Higher is better.
- **CNNaffinity** (kcal/mol): Predicted binding affinity. More negative is better.
- **vina_score**: Standard AutoDock Vina score.

### Programmatic Usage

```python
from rescoring import GninaRescorer, GninaConfig, ScoreFilter

# Configure rescoring
config = GninaConfig(
    receptor_path='receptor.pdb',
    poses_path='docked_poses.sdf',
    output_dir='rescored',
    cnn_scoring='rescore',
    cnn_score_threshold=0.9
)

# Rescore
rescorer = GninaRescorer(config)
results = rescorer.rescore()

# Filter
filtered = rescorer.filter_by_cnn_score(results, threshold=0.9)
filtered = rescorer.filter_by_cnn_affinity(filtered, threshold=-7.0)
```

### Multi-Criteria Filtering

```python
from rescoring import ScoreFilter

filter = ScoreFilter(logic='AND')
filter.add_threshold('CNNscore', 0.9, keep_above=True)
filter.add_threshold('CNNaffinity', -7.0, keep_above=False)
filter.add_threshold('vina_score', -8.0, keep_above=False)

result = filter.apply(rescoring_results)
print(f"Passed: {len(result.passed)}, Failed: {len(result.failed)}")
```

### Result Aggregation

```python
from rescoring import ResultAggregator

aggregator = ResultAggregator()
for cluster_name, results in all_results.items():
    aggregator.add_results(cluster_name, results)

# Analysis
best_per_ligand = aggregator.get_best_per_ligand()
consensus = aggregator.get_consensus_binders(min_clusters=3)

# Export
aggregator.to_csv('results.csv')
aggregator.to_json('results.json')
df = aggregator.to_dataframe()  # Requires pandas
```

## Extending the Pipeline

### Adding a New Rescoring Method

1. Create your rescorer class inheriting from `RescoringEngine`:

```python
# rescoring/my_rescorer.py
from .base import RescoringEngine, RescoringConfig, RescoringResult

class MyRescorerConfig(RescoringConfig):
    custom_param: float = 1.0

class MyRescorer(RescoringEngine):
    def rescore(self, poses=None):
        # Your implementation
        pass
    
    def rescore_file(self, pose_file):
        # Your implementation
        pass
```

2. Register in `run_rescoring.py`:

```python
RESCORING_METHODS = {
    'gnina': {...},
    'my_method': {
        'class': MyRescorer,
        'config_class': MyRescorerConfig,
        'description': 'My custom rescoring method',
        'scores': ['my_score']
    },
}
```

3. Use via CLI:

```bash
python run_rescoring.py --method my_method ...
```

See `rescoring/custom_template.py` for a complete template.

## Output Files

### Docking Output

```
docking_output/
├── cluster_0/
│   ├── prepared/           # Prepared receptor PDBQT
│   ├── prepared_ligands/   # Prepared ligand PDBQTs
│   ├── ligand1_out.pdbqt   # Docked poses
│   └── ligand2_out.pdbqt
├── cluster_1/
│   └── ...
└── docking_stats.json      # Docking statistics
```

### Rescoring Output

```
rescoring_output/
├── cluster_0/
│   ├── ligand1_gnina.sdf   # Rescored poses with CNN scores
│   └── ligand2_gnina.sdf
├── filtered_results.csv    # All filtered results
├── filtered_results.json   # JSON format results
├── filtered_poses.sdf      # High-quality poses for training
└── rescoring_stats.json    # Summary statistics
```

## Performance Tuning

### Uni-Dock

- **gpu_batch_size**: Start with 128, increase if GPU memory allows (256, 512)
- **exhaustiveness**: Use 128 for production, 32 for testing
- **search_mode**: 'fast' for initial screens, 'detail' for final docking

### Gnina

- **cnn_scoring**: 'rescore' is fastest, 'refinement' for better poses
- **cpu**: Increase for parallel CNN inference
- **no_gpu**: Use if GPU memory is limited

## Recommended Thresholds

For generating high-quality training data for FLOWR.ROOT:

| Score | Threshold | Rationale |
|-------|-----------|-----------|
| CNNscore | ≥ 0.9 | High confidence in bioactivity |
| CNNaffinity | ≤ -7.0 | ~μM binding (pK = 7) |
| Vina score | ≤ -7.0 | Reasonable binding energy |

## Troubleshooting

### Common Issues

1. **"Uni-Dock not found"**: Install Uni-Dock and ensure it's in PATH
2. **"Gnina not found"**: Install Gnina from https://github.com/gnina/gnina
3. **"RDKit not available"**: `pip install rdkit` or `conda install -c conda-forge rdkit`
4. **"Failed to convert receptor"**: Install Open Babel or ADFRsuite

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## References

- Uni-Dock: https://github.com/dptech-corp/Uni-Dock
- Gnina: https://github.com/gnina/gnina
- AutoDock Vina: https://vina.scripps.edu/
