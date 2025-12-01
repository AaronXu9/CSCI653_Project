#!/usr/bin/env python3
"""
Complete ensemble docking and rescoring pipeline.

This script provides a unified interface for running the complete
synthetic training data generation pipeline:
1. Dock ligand library against BioEmu cluster representatives
2. Rescore poses with CNN-based scoring
3. Filter high-quality poses for FLOWR.ROOT training

Usage:
    python run_pipeline.py --cluster_dir bioemu_clusters \
                           --ligand_library ligands.sdf \
                           --output_dir pipeline_output \
                           --center 10.5 22.1 -5.4
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from docking import UniDockEngine, UniDockConfig, detect_binding_site
from rescoring import GninaRescorer, GninaConfig, ScoreFilter, ResultAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Complete ensemble docking and rescoring pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--cluster_dir', type=Path, required=True,
                        help='Directory containing cluster_*.pdb files')
    parser.add_argument('--ligand_library', type=Path, required=True,
                        help='Ligand library (SDF file or directory)')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='Output directory for all results')
    
    # Docking box
    parser.add_argument('--center', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Docking box center (required if no reference ligand)')
    parser.add_argument('--size', type=float, nargs=3, default=[20, 20, 20],
                        metavar=('X', 'Y', 'Z'), help='Docking box size')
    parser.add_argument('--reference_ligand', type=Path,
                        help='Reference ligand for automatic box detection')
    
    # Pipeline options
    parser.add_argument('--skip_docking', action='store_true',
                        help='Skip docking, only run rescoring')
    parser.add_argument('--skip_rescoring', action='store_true',
                        help='Skip rescoring, only run docking')
    parser.add_argument('--n_clusters', type=int,
                        help='Number of clusters to process (default: all)')
    
    # Docking parameters
    parser.add_argument('--exhaustiveness', type=int, default=128)
    parser.add_argument('--num_modes', type=int, default=10)
    parser.add_argument('--gpu_batch_size', type=int, default=128)
    
    # Rescoring parameters
    parser.add_argument('--cnn_score_threshold', type=float, default=0.9,
                        help='Minimum CNN score (0-1)')
    parser.add_argument('--cnn_affinity_threshold', type=float, default=-7.0,
                        help='Maximum CNN affinity (kcal/mol)')
    
    return parser.parse_args()


def run_docking_phase(args, clusters: list) -> Dict[str, Any]:
    """Run the docking phase.
    
    Args:
        args: Command line arguments
        clusters: List of cluster PDB paths
        
    Returns:
        Dictionary with docking statistics
    """
    logger.info("="*60)
    logger.info("PHASE 1: HIGH-THROUGHPUT DOCKING (Uni-Dock)")
    logger.info("="*60)
    
    docking_output = args.output_dir / 'docking'
    docking_output.mkdir(parents=True, exist_ok=True)
    
    # Determine docking box
    if args.center:
        center = tuple(args.center)
        size = tuple(args.size)
    elif args.reference_ligand:
        from docking.utils import get_box_from_reference_ligand
        center, size = get_box_from_reference_ligand(args.reference_ligand)
        logger.info(f"Box from reference ligand: center={center}, size={size}")
    else:
        # Use first cluster to determine box
        center, size = detect_binding_site(clusters[0], method='geometric_center')
        logger.info(f"Auto-detected box: center={center}, size={size}")
    
    stats = {
        'n_clusters': len(clusters),
        'box_center': center,
        'box_size': size,
        'ligand_library': str(args.ligand_library),
        'per_cluster': []
    }
    
    for cluster_pdb in clusters:
        cluster_name = cluster_pdb.stem
        cluster_output = docking_output / cluster_name
        
        logger.info(f"\nDocking {cluster_name}...")
        
        config = UniDockConfig(
            receptor_path=cluster_pdb,
            ligand_paths=args.ligand_library,
            output_dir=cluster_output,
            center_x=center[0],
            center_y=center[1],
            center_z=center[2],
            size_x=size[0],
            size_y=size[1],
            size_z=size[2],
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            gpu_batch_size=args.gpu_batch_size
        )
        
        try:
            engine = UniDockEngine(config)
            results = engine.dock()
            
            cluster_stats = {
                'cluster': cluster_name,
                'status': 'success',
                'n_ligands': len(results),
                'n_poses': sum(r.num_poses for r in results)
            }
            logger.info(f"  Completed: {len(results)} ligands")
            
        except Exception as e:
            cluster_stats = {
                'cluster': cluster_name,
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"  Failed: {e}")
        
        stats['per_cluster'].append(cluster_stats)
    
    # Save docking stats
    with open(docking_output / 'docking_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def run_rescoring_phase(args, clusters: list) -> Dict[str, Any]:
    """Run the rescoring phase.
    
    Args:
        args: Command line arguments
        clusters: List of cluster PDB paths
        
    Returns:
        Dictionary with rescoring statistics
    """
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: CNN RESCORING (Gnina)")
    logger.info("="*60)
    
    docking_output = args.output_dir / 'docking'
    rescoring_output = args.output_dir / 'rescoring'
    rescoring_output.mkdir(parents=True, exist_ok=True)
    
    aggregator = ResultAggregator()
    
    for cluster_pdb in clusters:
        cluster_name = cluster_pdb.stem
        cluster_docking = docking_output / cluster_name
        
        if not cluster_docking.exists():
            logger.warning(f"No docking output for {cluster_name}, skipping")
            continue
        
        # Find pose files
        pose_files = list(cluster_docking.glob('*_out.sdf')) + \
                     list(cluster_docking.glob('*_out.pdbqt'))
        
        if not pose_files:
            logger.warning(f"No pose files for {cluster_name}")
            continue
        
        logger.info(f"\nRescoring {cluster_name}: {len(pose_files)} files")
        
        cluster_results = []
        for pose_file in pose_files:
            try:
                config = GninaConfig(
                    receptor_path=cluster_pdb,
                    poses_path=pose_file,
                    output_dir=rescoring_output / cluster_name,
                    cnn_scoring='rescore',
                    cnn_score_threshold=args.cnn_score_threshold,
                    cnn_affinity_threshold=args.cnn_affinity_threshold
                )
                
                rescorer = GninaRescorer(config)
                results = rescorer.rescore_file(pose_file)
                
                # Apply filters
                filtered = rescorer.filter_by_cnn_score(results, args.cnn_score_threshold)
                filtered = rescorer.filter_by_cnn_affinity(filtered, args.cnn_affinity_threshold)
                
                cluster_results.extend(filtered)
                
            except Exception as e:
                logger.error(f"  Error rescoring {pose_file}: {e}")
        
        aggregator.add_results(cluster_name, cluster_results)
        logger.info(f"  Passed filters: {len(cluster_results)} poses")
    
    # Export results
    aggregator.to_csv(rescoring_output / 'filtered_results.csv')
    aggregator.to_json(rescoring_output / 'filtered_results.json')
    
    summary = aggregator.summary()
    
    # Save summary
    with open(rescoring_output / 'rescoring_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    args = parse_args()
    
    logger.info("="*60)
    logger.info("ENSEMBLE DOCKING PIPELINE FOR FLOWR.ROOT TRAINING DATA")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    
    # Validate inputs
    if not args.cluster_dir.exists():
        logger.error(f"Cluster directory not found: {args.cluster_dir}")
        return 1
    
    if not args.ligand_library.exists():
        logger.error(f"Ligand library not found: {args.ligand_library}")
        return 1
    
    if not args.center and not args.reference_ligand:
        logger.warning("No docking box specified - will use geometric center")
    
    # Get cluster files
    clusters = sorted(args.cluster_dir.glob('cluster_*.pdb'))
    if args.n_clusters:
        clusters = clusters[:args.n_clusters]
    
    if not clusters:
        logger.error("No cluster files found")
        return 1
    
    logger.info(f"Processing {len(clusters)} clusters")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler for logging
    fh = logging.FileHandler(args.output_dir / 'pipeline.log')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)
    
    # Run phases
    docking_stats = None
    rescoring_stats = None
    
    if not args.skip_docking:
        docking_stats = run_docking_phase(args, clusters)
    
    if not args.skip_rescoring:
        rescoring_stats = run_rescoring_phase(args, clusters)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    
    if docking_stats:
        n_success = sum(1 for s in docking_stats['per_cluster'] if s['status'] == 'success')
        logger.info(f"Docking: {n_success}/{len(clusters)} clusters successful")
    
    if rescoring_stats:
        logger.info(f"Rescoring: {rescoring_stats['n_passed']} poses passed filters")
        logger.info(f"Unique ligands: {rescoring_stats['n_unique_ligands']}")
    
    logger.info(f"\nOutput directory: {args.output_dir}")
    logger.info(f"End time: {datetime.now().isoformat()}")
    
    # Save pipeline config
    config = {
        'cluster_dir': str(args.cluster_dir),
        'ligand_library': str(args.ligand_library),
        'n_clusters': len(clusters),
        'exhaustiveness': args.exhaustiveness,
        'num_modes': args.num_modes,
        'cnn_score_threshold': args.cnn_score_threshold,
        'cnn_affinity_threshold': args.cnn_affinity_threshold,
        'docking_stats': docking_stats,
        'rescoring_stats': rescoring_stats
    }
    
    with open(args.output_dir / 'pipeline_config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
