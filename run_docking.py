#!/usr/bin/env python3
"""
Batch docking script for BioEmu cluster ensemble docking.

This script performs high-throughput docking of a ligand library against
the BioEmu-generated cluster representatives using Uni-Dock.

Usage:
    python run_docking.py --cluster_dir bioemu_clusters \
                          --ligand_library ligands.sdf \
                          --output_dir docking_output \
                          --center 10.5 22.1 -5.4 \
                          --size 20 20 20

For SLURM submission, use the accompanying sbatch script.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from docking import UniDockEngine, UniDockConfig, detect_binding_site, get_box_from_reference_ligand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('docking.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch docking against BioEmu cluster representatives',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--cluster_dir', type=Path, required=True,
                        help='Directory containing cluster_*.pdb files')
    parser.add_argument('--ligand_library', type=Path, required=True,
                        help='Ligand library file (SDF, index file, or directory)')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='Output directory for docking results')
    
    # Docking box specification
    box_group = parser.add_argument_group('Docking Box')
    box_group.add_argument('--center', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                           help='Docking box center coordinates')
    box_group.add_argument('--size', type=float, nargs=3, default=[20, 20, 20],
                           metavar=('X', 'Y', 'Z'),
                           help='Docking box dimensions (Angstroms)')
    box_group.add_argument('--reference_ligand', type=Path,
                           help='Reference ligand for automatic box definition')
    box_group.add_argument('--pocket_residues', type=str,
                           help='Comma-separated residue IDs for pocket definition')
    box_group.add_argument('--padding', type=float, default=5.0,
                           help='Padding around detected binding site')
    
    # Uni-Dock parameters
    dock_group = parser.add_argument_group('Docking Parameters')
    dock_group.add_argument('--exhaustiveness', type=int, default=128,
                            help='Search exhaustiveness')
    dock_group.add_argument('--num_modes', type=int, default=10,
                            help='Number of poses to generate per ligand')
    dock_group.add_argument('--scoring', choices=['vina', 'vinardo', 'ad4'],
                            default='vina', help='Scoring function')
    dock_group.add_argument('--gpu_batch_size', type=int, default=128,
                            help='GPU batch size')
    dock_group.add_argument('--search_mode', choices=['fast', 'balance', 'detail'],
                            help='Search mode preset')
    dock_group.add_argument('--seed', type=int, help='Random seed')
    
    # Cluster selection
    cluster_group = parser.add_argument_group('Cluster Selection')
    cluster_group.add_argument('--cluster_ids', type=int, nargs='+',
                               help='Specific cluster IDs to dock (default: all)')
    cluster_group.add_argument('--n_clusters', type=int,
                               help='Number of clusters to dock (default: all)')
    
    # Execution options
    exec_group = parser.add_argument_group('Execution Options')
    exec_group.add_argument('--dry_run', action='store_true',
                            help='Print commands without executing')
    exec_group.add_argument('--skip_existing', action='store_true',
                            help='Skip clusters with existing output')
    exec_group.add_argument('--parallel_clusters', type=int, default=1,
                            help='Number of clusters to process in parallel')
    
    return parser.parse_args()


def get_cluster_files(cluster_dir: Path, 
                      cluster_ids: Optional[List[int]] = None,
                      n_clusters: Optional[int] = None) -> List[Path]:
    """Get list of cluster PDB files to process.
    
    Args:
        cluster_dir: Directory with cluster files
        cluster_ids: Specific cluster IDs to include
        n_clusters: Maximum number of clusters
        
    Returns:
        List of cluster file paths
    """
    all_clusters = sorted(cluster_dir.glob('cluster_*.pdb'))
    
    if cluster_ids:
        clusters = [c for c in all_clusters 
                    if int(c.stem.split('_')[1]) in cluster_ids]
    else:
        clusters = all_clusters
    
    if n_clusters:
        clusters = clusters[:n_clusters]
    
    return clusters


def determine_docking_box(args, cluster_pdb: Path) -> Tuple[Tuple[float, float, float], 
                                                            Tuple[float, float, float]]:
    """Determine docking box center and size.
    
    Args:
        args: Command line arguments
        cluster_pdb: Path to cluster PDB file
        
    Returns:
        Tuple of (center, size) coordinates
    """
    if args.center:
        center = tuple(args.center)
        size = tuple(args.size)
        return center, size
    
    if args.reference_ligand:
        logger.info(f"Detecting box from reference ligand: {args.reference_ligand}")
        return get_box_from_reference_ligand(args.reference_ligand, args.padding)
    
    if args.pocket_residues:
        residue_ids = [int(r.strip()) for r in args.pocket_residues.split(',')]
        logger.info(f"Detecting box from residues: {residue_ids}")
        return detect_binding_site(
            cluster_pdb, 
            method='residues',
            residue_ids=residue_ids,
            padding=args.padding
        )
    
    # Default: use geometric center of CA atoms
    logger.warning("No docking box specified, using geometric center of protein")
    return detect_binding_site(cluster_pdb, method='geometric_center', padding=args.padding)


def dock_single_cluster(cluster_pdb: Path,
                        ligand_library: Path,
                        output_dir: Path,
                        center: Tuple[float, float, float],
                        size: Tuple[float, float, float],
                        args) -> dict:
    """Dock ligand library against a single cluster.
    
    Args:
        cluster_pdb: Path to cluster PDB file
        ligand_library: Path to ligand library
        output_dir: Output directory for this cluster
        center: Docking box center
        size: Docking box size
        args: Additional arguments
        
    Returns:
        Dictionary with docking statistics
    """
    cluster_name = cluster_pdb.stem
    cluster_output = output_dir / cluster_name
    
    logger.info(f"Processing {cluster_name}")
    logger.info(f"  Receptor: {cluster_pdb}")
    logger.info(f"  Box center: {center}")
    logger.info(f"  Box size: {size}")
    
    # Check for existing output
    if args.skip_existing and cluster_output.exists():
        existing_outputs = list(cluster_output.glob('*_out.*'))
        if existing_outputs:
            logger.info(f"  Skipping - {len(existing_outputs)} outputs already exist")
            return {'cluster': cluster_name, 'status': 'skipped', 'n_outputs': len(existing_outputs)}
    
    # Create configuration
    config = UniDockConfig(
        receptor_path=cluster_pdb,
        ligand_paths=ligand_library,
        output_dir=cluster_output,
        center_x=center[0],
        center_y=center[1],
        center_z=center[2],
        size_x=size[0],
        size_y=size[1],
        size_z=size[2],
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        scoring=args.scoring,
        gpu_batch_size=args.gpu_batch_size,
        search_mode=args.search_mode,
        seed=args.seed
    )
    
    if args.dry_run:
        engine = UniDockEngine(config)
        engine.prepare_receptor()
        engine.prepare_ligands()
        cmd = engine._build_command()
        logger.info(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
        return {'cluster': cluster_name, 'status': 'dry_run', 'command': ' '.join(cmd)}
    
    # Execute docking
    try:
        engine = UniDockEngine(config)
        results = engine.dock()
        
        stats = {
            'cluster': cluster_name,
            'status': 'success',
            'n_ligands': len(results),
            'n_poses': sum(r.num_poses for r in results),
            'best_score': min((r.best_score for r in results if r.best_score), default=None)
        }
        
        logger.info(f"  Completed: {stats['n_ligands']} ligands, {stats['n_poses']} poses")
        if stats['best_score']:
            logger.info(f"  Best score: {stats['best_score']:.2f} kcal/mol")
        
        return stats
        
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return {'cluster': cluster_name, 'status': 'error', 'error': str(e)}


def main():
    args = parse_args()
    
    # Validate inputs
    if not args.cluster_dir.exists():
        logger.error(f"Cluster directory not found: {args.cluster_dir}")
        sys.exit(1)
    
    if not args.ligand_library.exists():
        logger.error(f"Ligand library not found: {args.ligand_library}")
        sys.exit(1)
    
    # Get cluster files
    clusters = get_cluster_files(args.cluster_dir, args.cluster_ids, args.n_clusters)
    if not clusters:
        logger.error("No cluster files found")
        sys.exit(1)
    
    logger.info(f"Found {len(clusters)} clusters to process")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process clusters
    all_stats = []
    
    for cluster_pdb in clusters:
        # Determine docking box (may differ per cluster if using pocket residues)
        center, size = determine_docking_box(args, cluster_pdb)
        
        stats = dock_single_cluster(
            cluster_pdb=cluster_pdb,
            ligand_library=args.ligand_library,
            output_dir=args.output_dir,
            center=center,
            size=size,
            args=args
        )
        all_stats.append(stats)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DOCKING SUMMARY")
    logger.info("="*60)
    
    n_success = sum(1 for s in all_stats if s['status'] == 'success')
    n_skipped = sum(1 for s in all_stats if s['status'] == 'skipped')
    n_error = sum(1 for s in all_stats if s['status'] == 'error')
    
    logger.info(f"Total clusters: {len(clusters)}")
    logger.info(f"  Successful: {n_success}")
    logger.info(f"  Skipped: {n_skipped}")
    logger.info(f"  Failed: {n_error}")
    
    if n_success > 0:
        total_ligands = sum(s.get('n_ligands', 0) for s in all_stats if s['status'] == 'success')
        total_poses = sum(s.get('n_poses', 0) for s in all_stats if s['status'] == 'success')
        logger.info(f"Total ligands docked: {total_ligands}")
        logger.info(f"Total poses generated: {total_poses}")
    
    # Save statistics
    stats_file = args.output_dir / 'docking_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Statistics saved to: {stats_file}")
    
    return 0 if n_error == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
