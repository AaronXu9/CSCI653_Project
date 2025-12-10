"""
Gnina CNN-based rescoring engine.

Gnina uses 3D convolutional neural networks to rescore docking poses,
providing superior discrimination between true binders and decoys.

Reference: https://github.com/gnina/gnina
"""

import subprocess
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import csv

from .base import RescoringEngine, RescoringConfig, RescoringResult, PoseData

logger = logging.getLogger(__name__)


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    logger.debug(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


@dataclass
class GninaConfig(RescoringConfig):
    """Configuration for Gnina rescoring.
    
    Attributes:
        cnn_scoring: CNN scoring mode ('none', 'rescore', 'refinement', 'all')
        cnn_model: Path to custom CNN model (None = default)
        cnn_weights: Path to CNN weights file (None = default)
        cnn_rotation: Number of CNN rotations for averaging
        minimize: Whether to minimize poses before scoring
        flex_residues: Flexible receptor residues (e.g., 'A:TYR123,A:PHE456')
        autobox_ligand: Ligand for automatic box definition
        autobox_add: Extra padding for autobox
        cpu: Number of CPU threads
        no_gpu: Disable GPU acceleration
    """
    cnn_scoring: str = 'rescore'
    cnn_model: Optional[Path] = None
    cnn_weights: Optional[Path] = None
    cnn_rotation: int = 0
    minimize: bool = False
    flex_residues: Optional[str] = None
    autobox_ligand: Optional[Path] = None
    autobox_add: float = 4.0
    cpu: int = 1
    no_gpu: bool = False
    
    # Score thresholds for filtering
    cnn_score_threshold: float = 0.0
    cnn_affinity_threshold: Optional[float] = None  # e.g., -7.0 for ~μM
    
    def __post_init__(self):
        super().__post_init__()
        valid_modes = ['none', 'rescore', 'refinement', 'all']
        if self.cnn_scoring not in valid_modes:
            raise ValueError(f"Invalid cnn_scoring: {self.cnn_scoring}. Must be one of {valid_modes}")


class GninaRescorer(RescoringEngine):
    """Gnina CNN-based rescoring engine.
    
    Gnina provides CNN-based scoring that is superior to traditional
    scoring functions for distinguishing true binding poses from decoys.
    
    Key scores:
        - CNNscore: Probability that pose is a true binding pose (0-1)
        - CNNaffinity: Predicted binding affinity (pK units)
        - Vina score: Standard AutoDock Vina score
    
    Example:
        >>> config = GninaConfig(
        ...     receptor_path='receptor.pdb',
        ...     poses_path='docked_poses.sdf',
        ...     output_dir='rescored',
        ...     cnn_scoring='rescore',
        ...     cnn_score_threshold=0.9
        ... )
        >>> rescorer = GninaRescorer(config)
        >>> results = rescorer.rescore()
        >>> filtered = rescorer.filter_by_cnn_score(results, threshold=0.9)
    """
    
    EXECUTABLE = '/home/aoxu/projects/PoseBench/forks/GNINA/gnina'
    
    def __init__(self, config: GninaConfig):
        """Initialize Gnina rescorer.
        
        Args:
            config: GninaConfig object
        """
        super().__init__(config)
        self.config: GninaConfig = config
    
    def _validate_config(self) -> None:
        """Validate Gnina configuration."""
        if not self.config.receptor_path.exists():
            raise ValueError(f"Receptor not found: {self.config.receptor_path}")
        if not self.config.poses_path.exists():
            raise ValueError(f"Poses not found: {self.config.poses_path}")
    
    def check_installation(self) -> bool:
        """Check if Gnina is installed."""
        try:
            result = run_command([self.EXECUTABLE, '--help'], check=False)
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("Gnina not found in PATH")
            return False
    
    def get_version(self) -> Optional[str]:
        """Get Gnina version."""
        try:
            result = run_command([self.EXECUTABLE, '--version'], check=False)
            if result.stdout:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def get_score_names(self) -> List[str]:
        """Get names of scores computed by Gnina."""
        return ['CNNscore', 'CNNaffinity', 'vina_score']
    
    def _build_command(self, 
                       pose_file: Path, 
                       output_file: Path,
                       score_only: bool = True) -> List[str]:
        """Build Gnina command line.
        
        Args:
            pose_file: Input file with poses
            output_file: Output file path
            score_only: If True, only score (no docking search)
            
        Returns:
            Command as list of strings
        """
        cmd = [self.EXECUTABLE]
        
        # Receptor
        cmd.extend(['-r', str(self.config.receptor_path)])
        
        # Ligand/poses
        cmd.extend(['-l', str(pose_file)])
        
        # Output
        cmd.extend(['-o', str(output_file)])
        
        # Score only mode (no search)
        if score_only:
            cmd.append('--score_only')
        
        # CNN scoring mode
        cmd.extend(['--cnn_scoring', self.config.cnn_scoring])
        
        # Custom model
        if self.config.cnn_model:
            cmd.extend(['--cnn', str(self.config.cnn_model)])
        if self.config.cnn_weights:
            cmd.extend(['--cnn_weights', str(self.config.cnn_weights)])
        
        # CNN rotations
        if self.config.cnn_rotation > 0:
            cmd.extend(['--cnn_rotation', str(self.config.cnn_rotation)])
        
        # Minimization
        if self.config.minimize:
            cmd.append('--minimize')
        
        # Autobox from ligand
        if self.config.autobox_ligand:
            cmd.extend(['--autobox_ligand', str(self.config.autobox_ligand)])
            cmd.extend(['--autobox_add', str(self.config.autobox_add)])
        
        # Flexible residues
        if self.config.flex_residues:
            cmd.extend(['--flexres', self.config.flex_residues])
        
        # CPU/GPU settings
        cmd.extend(['--cpu', str(self.config.cpu)])
        if self.config.no_gpu:
            cmd.append('--no_gpu')
        
        return cmd
    
    def rescore(self, poses: Optional[List[PoseData]] = None) -> List[RescoringResult]:
        """Rescore poses using Gnina CNN.
        
        Args:
            poses: Optional list of PoseData. If None, uses config.poses_path.
            
        Returns:
            List of RescoringResult objects
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        if poses is None:
            # Rescore all poses in the configured file/directory
            return self.rescore_file(self.config.poses_path)
        
        # Write poses to temporary file and rescore
        # This would require serialization of PoseData to SDF/PDBQT
        # For now, assume poses have file paths
        all_results = []
        for pose in poses:
            results = self.rescore_file(pose.pose_file)
            all_results.extend(results)
        
        return all_results
    
    def rescore_file(self, pose_file: Path) -> List[RescoringResult]:
        """Rescore all poses in a file.
        
        Args:
            pose_file: Path to file with poses
            
        Returns:
            List of RescoringResult objects
        """
        pose_file = Path(pose_file)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.config.output_dir / f"{pose_file.stem}_gnina.sdf"
        
        # Build and run command
        cmd = self._build_command(pose_file, output_file, score_only=True)
        logger.info(f"Running Gnina: {' '.join(cmd)}")
        
        try:
            result = run_command(cmd, check=True)
            logger.info("Gnina rescoring completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Gnina failed: {e.stderr}")
            raise RuntimeError(f"Gnina rescoring failed: {e}")
        
        # Parse output
        results = self._parse_output(output_file, pose_file)
        return results
    
    def _parse_output(self, output_file: Path, 
                      original_file: Path) -> List[RescoringResult]:
        """Parse Gnina output SDF file.
        
        Args:
            output_file: Path to Gnina output SDF
            original_file: Path to original pose file
            
        Returns:
            List of RescoringResult objects
        """
        results = []
        
        try:
            from rdkit import Chem
        except ImportError:
            logger.warning("RDKit not available, using text parsing")
            return self._parse_output_text(output_file, original_file)
        
        suppl = Chem.SDMolSupplier(str(output_file))
        receptor_name = self.config.receptor_path.stem
        
        for idx, mol in enumerate(suppl):
            if mol is None:
                continue
            
            # Get molecule name
            ligand_name = mol.GetProp('_Name') if mol.HasProp('_Name') else f"pose_{idx}"
            
            # Extract scores
            scores = {}
            cnn_score = None
            cnn_affinity = None
            vina_score = None
            
            # Try different property names Gnina might use
            score_props = {
                'CNNscore': ['CNNscore', 'CNN_score', 'cnn_score'],
                'CNNaffinity': ['CNNaffinity', 'CNN_affinity', 'cnn_affinity'],
                'vina_score': ['minimizedAffinity', 'vina_score', 'Affinity', 'affinity']
            }
            
            for score_name, prop_names in score_props.items():
                for prop in prop_names:
                    if mol.HasProp(prop):
                        try:
                            value = float(mol.GetProp(prop))
                            scores[score_name] = value
                            if score_name == 'CNNscore':
                                cnn_score = value
                            elif score_name == 'CNNaffinity':
                                cnn_affinity = value
                            elif score_name == 'vina_score':
                                vina_score = value
                            break
                        except ValueError:
                            pass
            
            # Create PoseData
            pose = PoseData(
                ligand_name=ligand_name,
                receptor_name=receptor_name,
                pose_file=output_file,
                pose_index=idx,
                original_score=vina_score
            )
            
            # Determine if pose passes thresholds
            passed = True
            if cnn_score is not None and self.config.cnn_score_threshold:
                passed = passed and (cnn_score >= self.config.cnn_score_threshold)
            if cnn_affinity is not None and self.config.cnn_affinity_threshold:
                passed = passed and (cnn_affinity <= self.config.cnn_affinity_threshold)
            
            result = RescoringResult(
                pose=pose,
                scores=scores,
                primary_score=cnn_score,
                confidence=cnn_score,
                passed_filter=passed,
                metadata={
                    'engine': 'Gnina',
                    'cnn_scoring_mode': self.config.cnn_scoring
                }
            )
            results.append(result)
        
        return results
    
    def _parse_output_text(self, output_file: Path, 
                           original_file: Path) -> List[RescoringResult]:
        """Parse Gnina output using text parsing (fallback without RDKit).
        
        Args:
            output_file: Path to output file
            original_file: Path to original file
            
        Returns:
            List of RescoringResult objects
        """
        results = []
        receptor_name = self.config.receptor_path.stem
        
        current_mol = {}
        mol_idx = 0
        
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('>  <'):
                    # Property line
                    prop_match = re.match(r'>  <(\w+)>', line)
                    if prop_match:
                        prop_name = prop_match.group(1)
                        value_line = next(f, '').strip()
                        try:
                            current_mol[prop_name] = float(value_line)
                        except ValueError:
                            current_mol[prop_name] = value_line
                
                elif line == '$$$$':
                    # End of molecule
                    if current_mol:
                        scores = {}
                        for key in ['CNNscore', 'CNNaffinity', 'minimizedAffinity']:
                            if key in current_mol:
                                scores[key] = current_mol[key]
                        
                        pose = PoseData(
                            ligand_name=current_mol.get('_Name', f'pose_{mol_idx}'),
                            receptor_name=receptor_name,
                            pose_file=output_file,
                            pose_index=mol_idx
                        )
                        
                        result = RescoringResult(
                            pose=pose,
                            scores=scores,
                            primary_score=scores.get('CNNscore'),
                            confidence=scores.get('CNNscore'),
                            metadata={'engine': 'Gnina'}
                        )
                        results.append(result)
                        
                        mol_idx += 1
                        current_mol = {}
        
        return results
    
    def filter_by_cnn_score(self, results: List[RescoringResult],
                            threshold: float = 0.9) -> List[RescoringResult]:
        """Filter results by CNN score threshold.
        
        Args:
            results: List of RescoringResult objects
            threshold: Minimum CNN score (0-1, higher = more confident)
            
        Returns:
            Filtered list of results
        """
        return self.filter_by_score(
            results, 
            threshold=threshold, 
            score_name='CNNscore',
            keep_above=True
        )
    
    def filter_by_cnn_affinity(self, results: List[RescoringResult],
                               threshold: float = -7.0) -> List[RescoringResult]:
        """Filter results by CNN affinity threshold.
        
        Args:
            results: List of RescoringResult objects
            threshold: Maximum CNN affinity (more negative = tighter binding)
            
        Returns:
            Filtered list of results
        """
        return self.filter_by_score(
            results,
            threshold=threshold,
            score_name='CNNaffinity', 
            keep_above=False  # Lower (more negative) is better
        )
    
    def get_top_poses(self, results: List[RescoringResult],
                      n: int = 1,
                      per_ligand: bool = True) -> List[RescoringResult]:
        """Get top-scoring poses.
        
        Args:
            results: List of RescoringResult objects
            n: Number of top poses to return
            per_ligand: If True, get top n poses per ligand
            
        Returns:
            List of top-scoring results
        """
        if not per_ligand:
            ranked = self.rank_poses(results, ascending=False)  # Higher CNN score is better
            return ranked[:n]
        
        # Group by ligand
        by_ligand = {}
        for result in results:
            ligand = result.ligand_name
            if ligand not in by_ligand:
                by_ligand[ligand] = []
            by_ligand[ligand].append(result)
        
        # Get top n per ligand
        top_poses = []
        for ligand, ligand_results in by_ligand.items():
            ranked = sorted(
                ligand_results,
                key=lambda r: r.primary_score if r.primary_score else 0,
                reverse=True
            )
            top_poses.extend(ranked[:n])
        
        return top_poses


def rescore_docking_results(
    docking_output_dir: Path,
    receptor_dir: Path,
    output_dir: Path,
    cnn_score_threshold: float = 0.9,
    cnn_affinity_threshold: float = -7.0,
    cnn_scoring: str = 'rescore'
) -> Dict[str, List[RescoringResult]]:
    """Rescore docking results from multiple clusters.
    
    Convenience function for rescoring Uni-Dock output against BioEmu clusters.
    
    Args:
        docking_output_dir: Directory with Uni-Dock outputs (per-cluster subdirs)
        receptor_dir: Directory with cluster PDB files
        output_dir: Output directory for rescored results
        cnn_score_threshold: Minimum CNN score to keep
        cnn_affinity_threshold: Maximum CNN affinity to keep
        cnn_scoring: Gnina CNN scoring mode
        
    Returns:
        Dictionary mapping cluster name to filtered RescoringResults
    """
    docking_output_dir = Path(docking_output_dir)
    receptor_dir = Path(receptor_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Find cluster subdirectories
    cluster_dirs = sorted([d for d in docking_output_dir.iterdir() if d.is_dir()])
    
    for cluster_out_dir in cluster_dirs:
        cluster_name = cluster_out_dir.name
        receptor_pdb = receptor_dir / f"{cluster_name}.pdb"
        
        if not receptor_pdb.exists():
            logger.warning(f"Receptor not found for {cluster_name}")
            continue
        
        # Find pose files
        pose_files = list(cluster_out_dir.glob('*_out.sdf')) + \
                     list(cluster_out_dir.glob('*_out.pdbqt'))
        
        if not pose_files:
            logger.warning(f"No pose files found in {cluster_out_dir}")
            continue
        
        logger.info(f"Rescoring {cluster_name}: {len(pose_files)} files")
        
        cluster_results = []
        for pose_file in pose_files:
            config = GninaConfig(
                receptor_path=receptor_pdb,
                poses_path=pose_file,
                output_dir=output_dir / cluster_name,
                cnn_scoring=cnn_scoring,
                cnn_score_threshold=cnn_score_threshold,
                cnn_affinity_threshold=cnn_affinity_threshold
            )
            
            rescorer = GninaRescorer(config)
            results = rescorer.rescore_file(pose_file)
            
            # Filter by thresholds
            filtered = rescorer.filter_by_cnn_score(results, cnn_score_threshold)
            filtered = rescorer.filter_by_cnn_affinity(filtered, cnn_affinity_threshold)
            
            cluster_results.extend(filtered)
        
        all_results[cluster_name] = cluster_results
        logger.info(f"{cluster_name}: {len(cluster_results)} poses passed filters")
    
    return all_results
