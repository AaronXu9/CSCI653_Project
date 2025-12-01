#!/usr/bin/env python3
"""
Rescoring script for CNN-based pose evaluation.

This script rescores docking poses from Uni-Dock using Gnina's CNN scoring
function, which provides superior discrimination between true binding poses
and decoys compared to traditional scoring functions.

Usage:
    python run_rescoring.py --docking_dir docking_output \
                            --cluster_dir bioemu_clusters \
                            --output_dir rescoring_output \
                            --cnn_score_threshold 0.9 \
                            --cnn_affinity_threshold -7.0

The script supports multiple rescoring methods through an extensible plugin system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rescoring import (
    GninaRescorer, GninaConfig,
    ScoreFilter, FilterResult, ResultAggregator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rescoring.log')
    ]
)
logger = logging.getLogger(__name__)


# Registry of available rescoring methods
RESCORING_METHODS = {
    'gnina': {
        'class': GninaRescorer,
        'config_class': GninaConfig,
        'description': 'Gnina CNN-based rescoring',
        'scores': ['CNNscore', 'CNNaffinity', 'vina_score']
    },
    # Future methods can be registered here:
    # 'rfscore': {'class': RFScoreRescorer, 'config_class': RFScoreConfig, ...},
    # 'onionnet': {'class': OnionNetRescorer, 'config_class': OnionNetConfig, ...},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rescore docking poses using CNN or other methods',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--docking_dir', type=Path, required=True,
                        help='Directory with docking outputs (per-cluster subdirs)')
    parser.add_argument('--cluster_dir', type=Path, required=True,
                        help='Directory with cluster PDB files')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='Output directory for rescoring results')
    
    # Rescoring method
    method_group = parser.add_argument_group('Rescoring Method')
    method_group.add_argument('--method', choices=list(RESCORING_METHODS.keys()),
                              default='gnina', help='Rescoring method to use')
    method_group.add_argument('--list_methods', action='store_true',
                              help='List available rescoring methods and exit')
    
    # Gnina-specific options
    gnina_group = parser.add_argument_group('Gnina Options')
    gnina_group.add_argument('--cnn_scoring', choices=['none', 'rescore', 'refinement', 'all'],
                             default='rescore', help='CNN scoring mode')
    gnina_group.add_argument('--cnn_model', type=Path, help='Custom CNN model file')
    gnina_group.add_argument('--cnn_weights', type=Path, help='Custom CNN weights file')
    gnina_group.add_argument('--minimize', action='store_true',
                             help='Minimize poses before scoring')
    gnina_group.add_argument('--cpu', type=int, default=1,
                             help='Number of CPU threads')
    gnina_group.add_argument('--no_gpu', action='store_true',
                             help='Disable GPU acceleration')
    
    # Filtering options
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group.add_argument('--cnn_score_threshold', type=float, default=0.9,
                              help='Minimum CNN score (0-1) to keep pose')
    filter_group.add_argument('--cnn_affinity_threshold', type=float, default=-7.0,
                              help='Maximum CNN affinity (more negative = tighter)')
    filter_group.add_argument('--vina_threshold', type=float,
                              help='Maximum Vina score threshold')
    filter_group.add_argument('--filter_logic', choices=['AND', 'OR'],
                              default='AND', help='Logic for combining filters')
    filter_group.add_argument('--top_n_per_ligand', type=int, default=1,
                              help='Keep top N poses per ligand')
    
    # Cluster selection
    cluster_group = parser.add_argument_group('Cluster Selection')
    cluster_group.add_argument('--cluster_ids', type=int, nargs='+',
                               help='Specific cluster IDs to rescore')
    cluster_group.add_argument('--n_clusters', type=int,
                               help='Number of clusters to rescore')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--export_csv', action='store_true',
                              help='Export results to CSV')
    output_group.add_argument('--export_json', action='store_true',
                              help='Export results to JSON')
    output_group.add_argument('--export_sdf', action='store_true',
                              help='Export filtered poses to SDF')
    output_group.add_argument('--keep_failed', action='store_true',
                              help='Also save poses that failed filters')
    
    return parser.parse_args()


def list_methods():
    """Print available rescoring methods."""
    print("\nAvailable Rescoring Methods:")
    print("="*60)
    for name, info in RESCORING_METHODS.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Scores: {', '.join(info['scores'])}")
    print()


def find_pose_files(docking_dir: Path, cluster_name: str) -> List[Path]:
    """Find docking output files for a cluster.
    
    Args:
        docking_dir: Base docking output directory
        cluster_name: Name of the cluster
        
    Returns:
        List of pose file paths
    """
    cluster_dir = docking_dir / cluster_name
    if not cluster_dir.exists():
        return []
    
    # Look for various output formats
    pose_files = []
    for pattern in ['*_out.sdf', '*_out.pdbqt', '*.sdf', '*.pdbqt']:
        pose_files.extend(cluster_dir.glob(pattern))
    
    # Deduplicate and sort
    pose_files = sorted(set(pose_files))
    return pose_files


def get_cluster_subdirs(docking_dir: Path,
                        cluster_ids: Optional[List[int]] = None,
                        n_clusters: Optional[int] = None) -> List[str]:
    """Get list of cluster subdirectories to process.
    
    Args:
        docking_dir: Base docking output directory
        cluster_ids: Specific cluster IDs to include
        n_clusters: Maximum number of clusters
        
    Returns:
        List of cluster names
    """
    all_clusters = sorted([d.name for d in docking_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('cluster_')])
    
    if cluster_ids:
        clusters = [c for c in all_clusters 
                    if int(c.split('_')[1]) in cluster_ids]
    else:
        clusters = all_clusters
    
    if n_clusters:
        clusters = clusters[:n_clusters]
    
    return clusters


def rescore_cluster(cluster_name: str,
                    pose_files: List[Path],
                    receptor_path: Path,
                    output_dir: Path,
                    args) -> Dict[str, Any]:
    """Rescore poses for a single cluster.
    
    Args:
        cluster_name: Name of the cluster
        pose_files: List of pose files to rescore
        receptor_path: Path to receptor PDB
        output_dir: Output directory
        args: Command line arguments
        
    Returns:
        Dictionary with rescoring statistics
    """
    logger.info(f"Rescoring {cluster_name}: {len(pose_files)} files")
    
    all_results = []
    
    for pose_file in pose_files:
        try:
            # Create Gnina config
            config = GninaConfig(
                receptor_path=receptor_path,
                poses_path=pose_file,
                output_dir=output_dir / cluster_name,
                cnn_scoring=args.cnn_scoring,
                cnn_model=args.cnn_model,
                cnn_weights=args.cnn_weights,
                minimize=args.minimize,
                cpu=args.cpu,
                no_gpu=args.no_gpu,
                cnn_score_threshold=args.cnn_score_threshold,
                cnn_affinity_threshold=args.cnn_affinity_threshold
            )
            
            rescorer = GninaRescorer(config)
            results = rescorer.rescore_file(pose_file)
            all_results.extend(results)
            
            logger.info(f"  {pose_file.name}: {len(results)} poses rescored")
            
        except Exception as e:
            logger.error(f"  Failed to rescore {pose_file}: {e}")
    
    # Apply filters
    score_filter = ScoreFilter(logic=args.filter_logic)
    score_filter.add_threshold('CNNscore', args.cnn_score_threshold, keep_above=True)
    score_filter.add_threshold('CNNaffinity', args.cnn_affinity_threshold, keep_above=False)
    
    if args.vina_threshold:
        score_filter.add_threshold('vina_score', args.vina_threshold, keep_above=False)
    
    filter_result = score_filter.apply(all_results)
    
    logger.info(f"  Filtering: {filter_result.summary()}")
    
    # Get top poses per ligand
    if args.top_n_per_ligand and filter_result.passed:
        # Group by ligand
        by_ligand = {}
        for result in filter_result.passed:
            ligand = result.ligand_name
            if ligand not in by_ligand:
                by_ligand[ligand] = []
            by_ligand[ligand].append(result)
        
        # Keep top N per ligand
        top_results = []
        for ligand, results in by_ligand.items():
            sorted_results = sorted(
                results,
                key=lambda r: r.scores.get('CNNscore', 0),
                reverse=True
            )
            top_results.extend(sorted_results[:args.top_n_per_ligand])
        
        filter_result.passed = top_results
        logger.info(f"  Top {args.top_n_per_ligand} per ligand: {len(top_results)} poses")
    
    return {
        'cluster': cluster_name,
        'n_files': len(pose_files),
        'n_total': len(all_results),
        'n_passed': len(filter_result.passed),
        'n_failed': len(filter_result.failed),
        'pass_rate': filter_result.pass_rate,
        'results': filter_result.passed,
        'failed': filter_result.failed if args.keep_failed else []
    }


def export_filtered_poses(results: List, output_path: Path):
    """Export filtered poses to SDF file.
    
    Args:
        results: List of RescoringResult objects
        output_path: Output SDF path
    """
    try:
        from rdkit import Chem
        
        writer = Chem.SDWriter(str(output_path))
        
        for result in results:
            # Load original pose file
            suppl = Chem.SDMolSupplier(str(result.pose.pose_file))
            mol = suppl[result.pose.pose_index] if suppl else None
            
            if mol:
                # Add scores as properties
                for score_name, score_value in result.scores.items():
                    mol.SetProp(score_name, str(score_value))
                mol.SetProp('receptor', result.receptor_name)
                writer.write(mol)
        
        writer.close()
        logger.info(f"Exported {len(results)} poses to {output_path}")
        
    except ImportError:
        logger.warning("RDKit not available, cannot export to SDF")


def main():
    args = parse_args()
    
    if args.list_methods:
        list_methods()
        return 0
    
    # Validate inputs
    if not args.docking_dir.exists():
        logger.error(f"Docking directory not found: {args.docking_dir}")
        sys.exit(1)
    
    if not args.cluster_dir.exists():
        logger.error(f"Cluster directory not found: {args.cluster_dir}")
        sys.exit(1)
    
    # Get clusters to process
    clusters = get_cluster_subdirs(args.docking_dir, args.cluster_ids, args.n_clusters)
    if not clusters:
        logger.error("No cluster directories found")
        sys.exit(1)
    
    logger.info(f"Found {len(clusters)} clusters to rescore")
    logger.info(f"Method: {args.method}")
    logger.info(f"Filters: CNNscore >= {args.cnn_score_threshold}, "
                f"CNNaffinity <= {args.cnn_affinity_threshold}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process clusters
    aggregator = ResultAggregator()
    all_stats = []
    
    for cluster_name in clusters:
        # Find receptor
        receptor_path = args.cluster_dir / f"{cluster_name}.pdb"
        if not receptor_path.exists():
            logger.warning(f"Receptor not found for {cluster_name}, skipping")
            continue
        
        # Find pose files
        pose_files = find_pose_files(args.docking_dir, cluster_name)
        if not pose_files:
            logger.warning(f"No pose files found for {cluster_name}, skipping")
            continue
        
        # Rescore
        stats = rescore_cluster(
            cluster_name=cluster_name,
            pose_files=pose_files,
            receptor_path=receptor_path,
            output_dir=args.output_dir,
            args=args
        )
        
        all_stats.append({k: v for k, v in stats.items() if k not in ['results', 'failed']})
        aggregator.add_results(cluster_name, stats['results'])
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("RESCORING SUMMARY")
    logger.info("="*60)
    
    summary = aggregator.summary()
    logger.info(f"Clusters processed: {summary['n_clusters']}")
    logger.info(f"Total poses rescored: {summary['n_total_results']}")
    logger.info(f"Poses passed filters: {summary['n_passed']}")
    logger.info(f"Unique ligands: {summary['n_unique_ligands']}")
    
    # Export results
    if args.export_csv:
        csv_path = args.output_dir / 'rescoring_results.csv'
        aggregator.to_csv(csv_path)
    
    if args.export_json:
        json_path = args.output_dir / 'rescoring_results.json'
        aggregator.to_json(json_path)
    
    if args.export_sdf:
        sdf_path = args.output_dir / 'filtered_poses.sdf'
        export_filtered_poses(aggregator.get_all_results(), sdf_path)
    
    # Save statistics
    stats_path = args.output_dir / 'rescoring_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'summary': summary,
            'per_cluster': all_stats,
            'parameters': {
                'method': args.method,
                'cnn_scoring': args.cnn_scoring,
                'cnn_score_threshold': args.cnn_score_threshold,
                'cnn_affinity_threshold': args.cnn_affinity_threshold,
                'filter_logic': args.filter_logic,
                'top_n_per_ligand': args.top_n_per_ligand
            }
        }, f, indent=2)
    logger.info(f"Statistics saved to: {stats_path}")
    
    # Best poses analysis
    best_poses = aggregator.get_best_per_ligand(score_name='CNNscore', higher_is_better=True)
    if best_poses:
        logger.info(f"\nTop {min(10, len(best_poses))} ligands by CNN score:")
        sorted_best = sorted(best_poses, key=lambda r: r.scores.get('CNNscore', 0), reverse=True)
        for i, result in enumerate(sorted_best[:10]):
            cnn_score = result.scores.get('CNNscore', 'N/A')
            cnn_aff = result.scores.get('CNNaffinity', 'N/A')
            logger.info(f"  {i+1}. {result.ligand_name}: CNNscore={cnn_score:.3f}, "
                       f"CNNaffinity={cnn_aff:.2f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
