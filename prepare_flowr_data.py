#!/usr/bin/env python3
"""
FLOWR.ROOT Data Preparation Script.

This script converts filtered docking poses into the LMDB format
required by FLOWR.ROOT for fine-tuning.

The pipeline:
    1. Export filtered poses to standardized directory structure
    2. Featurize protein and ligand structures (SE(3) features)
    3. Serialize features into LMDB database
    4. Compute data statistics for prior distribution

Usage:
    # From rescoring output
    python prepare_flowr_data.py \
        --rescoring_dir rescoring_output \
        --cluster_dir bioemu_clusters \
        --output_dir flowr_training_data \
        --cnn_score_threshold 0.9 \
        --cnn_affinity_threshold -7.0

    # From existing raw directory (resume)
    python prepare_flowr_data.py \
        --raw_dir flowr_training_data/raw \
        --output_dir flowr_training_data

    # Validation only
    python prepare_flowr_data.py \
        --validate flowr_training_data/final/custom_data.lmdb

Output Directory Structure:
    output_dir/
    ├── raw/                    # Exported pose files
    │   ├── system_0001/
    │   │   ├── system_0001_protein.pdb
    │   │   └── system_0001_ligand.sdf
    │   ├── system_0002/
    │   │   └── ...
    │   └── manifest.json
    ├── processed/              # Optional intermediate pickles
    └── final/
        ├── custom_data.lmdb    # Main database
        └── data_statistics.npz # Prior distribution stats
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from flowr_data import (
    FlowrDataPreparer,
    FlowrDataConfig,
    PoseExporter,
    ExportConfig,
    LMDBWriter,
    DataStatistics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('flowr_data_prep.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare training data for FLOWR.ROOT fine-tuning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input sources (mutually exclusive groups)
    input_group = parser.add_argument_group('Input Sources')
    input_group.add_argument('--rescoring_dir', type=Path,
                             help='Directory with rescoring outputs')
    input_group.add_argument('--rescoring_json', type=Path,
                             help='JSON file with rescoring results')
    input_group.add_argument('--docking_dir', type=Path,
                             help='Directory with raw docking outputs')
    input_group.add_argument('--raw_dir', type=Path,
                             help='Existing raw directory to resume from')
    input_group.add_argument('--cluster_dir', type=Path,
                             help='Directory with cluster PDB files')
    
    # Score file and filtering
    filter_group = parser.add_argument_group('Filtering')
    filter_group.add_argument('--score_file', type=Path,
                              help='CSV/JSON with scores for filtering')
    filter_group.add_argument('--cnn_score_threshold', type=float, default=0.9,
                              help='Minimum CNN score (0-1)')
    filter_group.add_argument('--cnn_affinity_threshold', type=float, default=-7.0,
                              help='Maximum CNN affinity (more negative = tighter)')
    filter_group.add_argument('--top_n_per_cluster', type=int,
                              help='Take top N poses per cluster')
    
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_dir', type=Path, default='flowr_training_data',
                              help='Base output directory')
    output_group.add_argument('--db_name', default='custom_data',
                              help='Name for LMDB database')
    
    # Featurization options
    feat_group = parser.add_argument_group('Featurization')
    feat_group.add_argument('--pocket_radius', type=float, default=10.0,
                            help='Radius (Å) around ligand for pocket extraction')
    feat_group.add_argument('--knn_k', type=int, default=16,
                            help='K for KNN graph construction')
    feat_group.add_argument('--use_hydrogens', action='store_true',
                            help='Include hydrogen atoms')
    feat_group.add_argument('--no_center', action='store_true',
                            help='Do not center coordinates on ligand')
    
    # Processing options
    proc_group = parser.add_argument_group('Processing')
    proc_group.add_argument('--parallel', action='store_true',
                            help='Use parallel processing')
    proc_group.add_argument('--n_workers', type=int, default=4,
                            help='Number of parallel workers')
    proc_group.add_argument('--save_pickles', action='store_true',
                            help='Save intermediate pickle files')
    proc_group.add_argument('--overwrite', action='store_true',
                            help='Overwrite existing data')
    
    # Validation
    val_group = parser.add_argument_group('Validation')
    val_group.add_argument('--validate', type=Path,
                           help='Validate existing LMDB and exit')
    val_group.add_argument('--validate_samples', type=int, default=10,
                           help='Number of samples to validate')
    
    # Additional options
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without executing')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def load_rescoring_results(rescoring_dir: Path,
                           score_threshold: float = 0.9,
                           affinity_threshold: float = -7.0,
                           top_n: Optional[int] = None
                           ) -> List[Dict[str, Any]]:
    """Load and filter rescoring results from directory.
    
    Args:
        rescoring_dir: Directory with rescoring outputs
        score_threshold: Minimum CNN score
        affinity_threshold: Maximum CNN affinity
        top_n: Top N per cluster
        
    Returns:
        List of filtered result dicts
    """
    results = []
    
    # Find all result files
    for cluster_dir in sorted(rescoring_dir.iterdir()):
        if not cluster_dir.is_dir():
            continue
        
        cluster_name = cluster_dir.name
        
        # Look for JSON results
        for json_file in cluster_dir.glob('*_rescored.json'):
            with open(json_file) as f:
                data = json.load(f)
            
            cluster_results = []
            for pose in data.get('poses', [data]):
                scores = pose.get('scores', pose)
                cnn_score = scores.get('CNNscore', 0)
                cnn_affinity = scores.get('CNNaffinity', 0)
                
                # Apply filters
                if cnn_score < score_threshold:
                    continue
                if cnn_affinity > affinity_threshold:
                    continue
                
                cluster_results.append({
                    'cluster': cluster_name,
                    'ligand': pose.get('ligand_name', json_file.stem),
                    'pose_file': pose.get('pose_file', str(cluster_dir / f"{json_file.stem}.sdf")),
                    'pose_index': pose.get('pose_index', 0),
                    'CNNscore': cnn_score,
                    'CNNaffinity': cnn_affinity,
                    'scores': scores
                })
            
            # Sort by affinity and take top N
            cluster_results.sort(key=lambda x: x['CNNaffinity'])
            if top_n:
                cluster_results = cluster_results[:top_n]
            
            results.extend(cluster_results)
    
    logger.info(f"Loaded {len(results)} filtered results from {rescoring_dir}")
    return results


def load_rescoring_json(json_path: Path,
                        score_threshold: float = 0.9,
                        affinity_threshold: float = -7.0
                        ) -> List[Dict[str, Any]]:
    """Load rescoring results from single JSON file.
    
    Args:
        json_path: Path to JSON file
        score_threshold: Minimum CNN score
        affinity_threshold: Maximum CNN affinity
        
    Returns:
        List of filtered result dicts
    """
    with open(json_path) as f:
        all_results = json.load(f)
    
    filtered = []
    for result in all_results:
        scores = result.get('scores', result)
        cnn_score = scores.get('CNNscore', 0)
        cnn_affinity = scores.get('CNNaffinity', 0)
        
        if cnn_score >= score_threshold and cnn_affinity <= affinity_threshold:
            filtered.append(result)
    
    logger.info(f"Loaded {len(filtered)} filtered results from {json_path}")
    return filtered


def build_cluster_pdb_map(cluster_dir: Path) -> Dict[str, Path]:
    """Build mapping from cluster names to PDB paths.
    
    Args:
        cluster_dir: Directory with cluster_*.pdb files
        
    Returns:
        Dict mapping cluster names to PDB paths
    """
    pdb_map = {}
    for pdb_file in cluster_dir.glob('cluster_*.pdb'):
        cluster_name = pdb_file.stem
        pdb_map[cluster_name] = pdb_file
    
    logger.info(f"Found {len(pdb_map)} cluster PDBs in {cluster_dir}")
    return pdb_map


def run_validation(lmdb_path: Path, n_samples: int = 10):
    """Run validation on existing LMDB.
    
    Args:
        lmdb_path: Path to LMDB file
        n_samples: Number of samples to check
    """
    from flowr_data.lmdb_writer import LMDBReader
    
    logger.info(f"Validating {lmdb_path}")
    
    reader = LMDBReader(lmdb_path)
    logger.info(f"Database contains {len(reader)} systems")
    
    # Check statistics file
    stats_path = lmdb_path.parent / 'data_statistics.npz'
    if stats_path.exists():
        stats = DataStatistics.load(stats_path)
        logger.info(f"Statistics: {stats.n_systems} systems")
        logger.info(f"  Affinity: mean={stats.affinity_stats.get('mean', 'N/A'):.2f}, "
                   f"std={stats.affinity_stats.get('std', 'N/A'):.2f}")
        logger.info(f"  Ligand atoms: mean={stats.ligand_atom_stats.get('mean', 'N/A'):.1f}")
        logger.info(f"  Protein atoms: mean={stats.protein_atom_stats.get('mean', 'N/A'):.1f}")
    
    # Validate samples
    import random
    indices = random.sample(range(len(reader)), min(n_samples, len(reader)))
    
    logger.info(f"\nValidating {len(indices)} random samples:")
    for idx in indices:
        features = reader[idx]
        logger.info(f"  {features.system_id}: "
                   f"protein={len(features.protein_pos)} atoms, "
                   f"ligand={len(features.ligand_pos)} atoms, "
                   f"affinity={features.affinity:.2f}")
        
        # Basic sanity checks
        assert features.protein_pos.shape[1] == 3
        assert features.ligand_pos.shape[1] == 3
        assert len(features.protein_elements) == len(features.protein_pos)
        assert len(features.ligand_elements) == len(features.ligand_pos)
    
    reader.close()
    logger.info("\nValidation passed!")


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validation mode
    if args.validate:
        run_validation(args.validate, args.validate_samples)
        return
    
    # Create configuration
    config = FlowrDataConfig(
        output_dir=args.output_dir,
        knn_k=args.knn_k,
        pocket_radius=args.pocket_radius,
        use_hydrogens=args.use_hydrogens,
        center_on_ligand=not args.no_center,
        copy_files=True,
        overwrite=args.overwrite,
        n_workers=args.n_workers
    )
    
    if args.dry_run:
        logger.info("DRY RUN - showing configuration:")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  Raw directory: {config.raw_dir}")
        logger.info(f"  Final directory: {config.final_dir}")
        logger.info(f"  Pocket radius: {config.pocket_radius} Å")
        logger.info(f"  KNN K: {config.knn_k}")
        logger.info(f"  Use hydrogens: {config.use_hydrogens}")
        logger.info(f"  Center on ligand: {config.center_on_ligand}")
        return
    
    # Initialize preparer
    preparer = FlowrDataPreparer(config)
    
    # Determine input source and process
    if args.raw_dir:
        # Resume from existing raw directory
        logger.info(f"Resuming from raw directory: {args.raw_dir}")
        stats = preparer.prepare_from_raw_directory(
            raw_dir=args.raw_dir,
            parallel=args.parallel
        )
    
    elif args.rescoring_dir or args.rescoring_json:
        # Load rescoring results
        if args.rescoring_json:
            results = load_rescoring_json(
                args.rescoring_json,
                score_threshold=args.cnn_score_threshold,
                affinity_threshold=args.cnn_affinity_threshold
            )
        else:
            results = load_rescoring_results(
                args.rescoring_dir,
                score_threshold=args.cnn_score_threshold,
                affinity_threshold=args.cnn_affinity_threshold,
                top_n=args.top_n_per_cluster
            )
        
        if not results:
            logger.error("No results passed filters!")
            sys.exit(1)
        
        # Build cluster PDB map
        if not args.cluster_dir:
            logger.error("--cluster_dir required with --rescoring_dir/--rescoring_json")
            sys.exit(1)
        
        cluster_pdb_map = build_cluster_pdb_map(args.cluster_dir)
        
        # Export and process
        # Note: We need to convert the dict results to the proper format
        # For now, use the directory-based export
        exported = []
        exporter = PoseExporter(config.to_export_config())
        
        for result in results:
            cluster_name = result['cluster']
            if cluster_name not in cluster_pdb_map:
                logger.warning(f"Cluster {cluster_name} not found in PDB map")
                continue
            
            system = exporter.export_pose(
                receptor_path=cluster_pdb_map[cluster_name],
                ligand_path=Path(result['pose_file']),
                affinity=result['CNNaffinity'],
                pose_index=result.get('pose_index', 0),
                cluster_id=cluster_name,
                ligand_name=result.get('ligand'),
                metadata={'scores': result.get('scores', {})}
            )
            
            if system:
                exported.append(system)
        
        exporter.save_manifest(exported)
        preparer._exported_systems = exported
        
        logger.info(f"Exported {len(exported)} systems")
        
        # Generate LMDB
        stats = preparer.generate_lmdb(
            exported,
            parallel=args.parallel,
            db_name=args.db_name
        )
    
    elif args.docking_dir:
        # Process from raw docking output
        if not args.cluster_dir:
            logger.error("--cluster_dir required with --docking_dir")
            sys.exit(1)
        
        exported = preparer.export_from_directory(
            docking_dir=args.docking_dir,
            cluster_dir=args.cluster_dir,
            score_file=args.score_file,
            score_threshold=args.cnn_affinity_threshold  # Use affinity as score
        )
        
        stats = preparer.generate_lmdb(
            exported,
            parallel=args.parallel,
            db_name=args.db_name
        )
    
    else:
        logger.error("Must specify one of: --rescoring_dir, --rescoring_json, "
                    "--docking_dir, or --raw_dir")
        sys.exit(1)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total systems: {stats.n_systems}")
    logger.info(f"LMDB path: {config.final_dir / f'{args.db_name}.lmdb'}")
    logger.info(f"Statistics: {config.final_dir / 'data_statistics.npz'}")
    
    if stats.affinity_stats:
        logger.info(f"\nAffinity distribution:")
        logger.info(f"  Mean: {stats.affinity_stats.get('mean', 'N/A'):.2f}")
        logger.info(f"  Std:  {stats.affinity_stats.get('std', 'N/A'):.2f}")
        logger.info(f"  Min:  {stats.affinity_stats.get('min', 'N/A'):.2f}")
        logger.info(f"  Max:  {stats.affinity_stats.get('max', 'N/A'):.2f}")
    
    if stats.ligand_atom_stats:
        logger.info(f"\nLigand atoms:")
        logger.info(f"  Mean: {stats.ligand_atom_stats.get('mean', 'N/A'):.1f}")
        logger.info(f"  Range: {stats.ligand_atom_stats.get('min', 'N/A')}-"
                   f"{stats.ligand_atom_stats.get('max', 'N/A')}")
    
    # Validate
    logger.info("\nRunning validation...")
    preparer.validate_lmdb()
    
    logger.info("\nNext steps:")
    logger.info("  1. Copy the LMDB to your FLOWR installation:")
    logger.info(f"     cp -r {config.final_dir} flowr/data/preprocess_data/custom_data/")
    logger.info("  2. Run the fine-tuning script (see finetune.sl)")


if __name__ == '__main__':
    main()
