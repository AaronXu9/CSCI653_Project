"""
Uni-Dock docking engine implementation.

Uni-Dock is a GPU-accelerated molecular docking program based on AutoDock Vina,
capable of extreme throughput for virtual screening campaigns.

Reference: https://github.com/dptech-corp/Uni-Dock
"""

import subprocess
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import glob
import re

from .base import DockingEngine, DockingConfig, DockingResult
from .utils import prepare_receptor_pdbqt, prepare_ligand_pdbqt, run_command

logger = logging.getLogger(__name__)


@dataclass
class UniDockConfig(DockingConfig):
    """Configuration for Uni-Dock docking engine.
    
    Extends base DockingConfig with Uni-Dock specific options.
    
    Attributes:
        scoring: Scoring function to use ('vina', 'vinardo', 'ad4')
        gpu_batch_size: Number of ligands to process per GPU batch
        search_mode: Search mode ('fast', 'balance', 'detail')
        refine_step: Number of refinement steps
        keep_nonpolar_H: Whether to keep non-polar hydrogens
        multi_bias: Enable multi-bias docking
        bias_file: Path to bias file for guided docking
    """
    scoring: str = 'vina'
    gpu_batch_size: int = 128
    search_mode: Optional[str] = None  # 'fast', 'balance', 'detail'
    refine_step: int = 5
    keep_nonpolar_H: bool = False
    multi_bias: bool = False
    bias_file: Optional[Path] = None
    
    # Uni-Dock specific batch options
    ligand_index: Optional[Path] = None  # Path to file listing ligands
    batch_ligand_dir: Optional[Path] = None  # Directory with ligands
    
    def __post_init__(self):
        super().__post_init__()
        if self.scoring not in ['vina', 'vinardo', 'ad4']:
            raise ValueError(f"Invalid scoring function: {self.scoring}")
        if self.search_mode and self.search_mode not in ['fast', 'balance', 'detail']:
            raise ValueError(f"Invalid search mode: {self.search_mode}")


class UniDockEngine(DockingEngine):
    """Uni-Dock docking engine for GPU-accelerated virtual screening.
    
    This engine wraps the Uni-Dock command-line tool for high-throughput
    docking of large ligand libraries against protein targets.
    
    Example:
        >>> config = UniDockConfig(
        ...     receptor_path='receptor.pdb',
        ...     ligand_paths=['ligand1.sdf', 'ligand2.sdf'],
        ...     output_dir='docking_output',
        ...     center_x=10.0, center_y=20.0, center_z=30.0,
        ...     exhaustiveness=128,
        ...     gpu_batch_size=256
        ... )
        >>> engine = UniDockEngine(config)
        >>> results = engine.dock()
    """
    
    EXECUTABLE = '/home/aoxu/miniconda3/envs/unidock2/bin/unidock'
    
    def __init__(self, config: UniDockConfig):
        """Initialize Uni-Dock engine.
        
        Args:
            config: UniDockConfig object with docking parameters
        """
        super().__init__(config)
        self.config: UniDockConfig = config
        self._prepared_receptor: Optional[Path] = None
        self._prepared_ligands: List[Path] = []
    
    def _validate_config(self) -> None:
        """Validate Uni-Dock configuration."""
        if not self.config.receptor_path.exists():
            raise ValueError(f"Receptor file not found: {self.config.receptor_path}")
        
        # Check ligand inputs
        if isinstance(self.config.ligand_paths, Path):
            if not self.config.ligand_paths.exists():
                raise ValueError(f"Ligand file/index not found: {self.config.ligand_paths}")
        elif isinstance(self.config.ligand_paths, list):
            for lp in self.config.ligand_paths:
                if not lp.exists():
                    raise ValueError(f"Ligand file not found: {lp}")
    
    def check_installation(self) -> bool:
        """Check if Uni-Dock is installed."""
        try:
            result = run_command([self.EXECUTABLE, '--help'], check=False)
            return True
        except FileNotFoundError:
            logger.warning("Uni-Dock not found in PATH")
            return False
    
    def get_version(self) -> Optional[str]:
        """Get Uni-Dock version."""
        try:
            result = run_command([self.EXECUTABLE, '--version'], check=False)
            # Parse version from output
            if result.stdout:
                match = re.search(r'[\d.]+', result.stdout)
                if match:
                    return match.group()
        except Exception:
            pass
        return None
    
    def prepare_receptor(self) -> Path:
        """Prepare receptor for docking.
        
        Uni-Dock supports PDB files directly, so we can skip PDBQT conversion
        if the input is already a PDB.
        
        Returns:
            Path to receptor file (PDB or PDBQT)
        """
        receptor_path = self.config.receptor_path
        
        # If it's already PDB or PDBQT, just use it
        if receptor_path.suffix.lower() in ['.pdb', '.pdbqt']:
            self._prepared_receptor = receptor_path
            return receptor_path
            
        # Otherwise convert to PDBQT (fallback)
        output_dir = self.config.output_dir / 'prepared'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{receptor_path.stem}.pdbqt"
        
        self._prepared_receptor = prepare_receptor_pdbqt(
            receptor_path, 
            output_path,
            add_hydrogens=True
        )
        return self._prepared_receptor
    
    def prepare_ligands(self) -> List[Path]:
        """Prepare ligands for docking.
        
        Returns:
            List of paths to ligand files (PDBQT or SDF)
        """
        output_dir = self.config.output_dir / 'prepared_ligands'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ligand_paths = self.config.ligand_paths
        if isinstance(ligand_paths, Path):
            # If it's an index file, read the paths
            if ligand_paths.suffix in ['.txt', '.lst', '.index']:
                with open(ligand_paths) as f:
                    ligand_paths = [Path(line.strip()) for line in f if line.strip()]
            else:
                ligand_paths = [ligand_paths]
        
        prepared = []
        for ligand_path in ligand_paths:
            try:
                # Uni-Dock supports SDF directly via ligand_index
                if ligand_path.suffix.lower() in ['.pdbqt', '.sdf']:
                    prepared.append(ligand_path)
                else:
                    output_path = output_dir / f"{ligand_path.stem}.pdbqt"
                    result = prepare_ligand_pdbqt(ligand_path, output_path)
                    prepared.append(result)
            except Exception as e:
                logger.warning(f"Failed to prepare ligand {ligand_path}: {e}")
        
        self._prepared_ligands = prepared
        return prepared
    
    def _build_command(self) -> List[str]:
        """Build Uni-Dock command line.
        
        Returns:
            Command as list of strings
        """
        cmd = [self.EXECUTABLE]
        
        # Receptor
        cmd.extend(['--receptor', str(self._prepared_receptor)])
        
        # Ligands - handle batch input
        if self.config.ligand_index:
            cmd.extend(['--ligand_index', str(self.config.ligand_index)])
        elif self.config.batch_ligand_dir:
            cmd.extend(['--dir', str(self.config.batch_ligand_dir)])
        else:
            # Always use ligand_index to support SDFs and batching
            index_file = self.config.output_dir / 'ligand_index.txt'
            with open(index_file, 'w') as f:
                for lig in self._prepared_ligands:
                    f.write(f"{lig}\n")
            cmd.extend(['--ligand_index', str(index_file)])
        
        # Docking box
        cmd.extend([
            '--center_x', str(self.config.center_x),
            '--center_y', str(self.config.center_y),
            '--center_z', str(self.config.center_z),
            '--size_x', str(self.config.size_x),
            '--size_y', str(self.config.size_y),
            '--size_z', str(self.config.size_z),
        ])
        
        # Search parameters
        cmd.extend(['--exhaustiveness', str(self.config.exhaustiveness)])
        cmd.extend(['--num_modes', str(self.config.num_modes)])
        cmd.extend(['--energy_range', str(self.config.energy_range)])
        
        # Scoring function
        cmd.extend(['--scoring', self.config.scoring])
        
        # GPU batch size - removed as it is not supported by the installed version
        # cmd.extend(['--gpu_batch_size', str(self.config.gpu_batch_size)])
        
        # Search mode
        if self.config.search_mode:
            cmd.extend(['--search_mode', self.config.search_mode])
        
        # Refinement
        cmd.extend(['--refine_step', str(self.config.refine_step)])
        
        # Output directory
        cmd.extend(['--dir', str(self.config.output_dir)])
        
        # Seed
        if self.config.seed is not None:
            cmd.extend(['--seed', str(self.config.seed)])
        
        return cmd
    
    def dock(self) -> List[DockingResult]:
        """Execute Uni-Dock docking.
        
        Returns:
            List of DockingResult objects for each ligand
        """
        # Prepare files
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self._prepared_receptor is None:
            self.prepare_receptor()
        
        if not self._prepared_ligands:
            self.prepare_ligands()
        
        logger.info(f"Starting Uni-Dock with {len(self._prepared_ligands)} ligands")
        
        # Build and run command
        cmd = self._build_command()
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = run_command(cmd, check=True)
            logger.info("Uni-Dock completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Uni-Dock failed: {e.stderr}")
            raise RuntimeError(f"Uni-Dock docking failed: {e}")
        
        # Parse results
        results = self._collect_results()
        return results
    
    def _collect_results(self) -> List[DockingResult]:
        """Collect and parse docking results.
        
        Returns:
            List of DockingResult objects
        """
        results = []
        output_dir = self.config.output_dir
        
        # Find output files (PDBQT or SDF)
        output_files = list(output_dir.glob('*_out.pdbqt')) + list(output_dir.glob('*_out.sdf'))
        
        for output_file in output_files:
            try:
                result = self.parse_output(output_file)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse output {output_file}: {e}")
        
        return results
    
    def parse_output(self, output_path: Path) -> DockingResult:
        """Parse Uni-Dock output file.
        
        Args:
            output_path: Path to output PDBQT or SDF file
            
        Returns:
            DockingResult object with scores
        """
        output_path = Path(output_path)
        scores = []
        
        if output_path.suffix.lower() == '.pdbqt':
            scores = self._parse_pdbqt_scores(output_path)
        elif output_path.suffix.lower() == '.sdf':
            scores = self._parse_sdf_scores(output_path)
        
        # Extract ligand name from filename
        ligand_name = output_path.stem.replace('_out', '')
        receptor_name = self._prepared_receptor.stem if self._prepared_receptor else 'unknown'
        
        return DockingResult(
            ligand_name=ligand_name,
            receptor_name=receptor_name,
            output_file=output_path,
            scores=scores,
            metadata={'engine': 'UniDock', 'scoring': self.config.scoring}
        )
    
    def _parse_pdbqt_scores(self, pdbqt_path: Path) -> List[float]:
        """Parse scores from PDBQT output file.
        
        Args:
            pdbqt_path: Path to PDBQT file
            
        Returns:
            List of docking scores
        """
        scores = []
        with open(pdbqt_path) as f:
            for line in f:
                if line.startswith('REMARK VINA RESULT:') or line.startswith('REMARK RESULT:'):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            score = float(parts[3])
                            scores.append(score)
                        except ValueError:
                            pass
        return scores
    
    def _parse_sdf_scores(self, sdf_path: Path) -> List[float]:
        """Parse scores from SDF output file.
        
        Args:
            sdf_path: Path to SDF file
            
        Returns:
            List of docking scores
        """
        scores = []
        try:
            from rdkit import Chem
            suppl = Chem.SDMolSupplier(str(sdf_path))
            for mol in suppl:
                if mol is None:
                    continue
                # Try different property names
                for prop in ['minimizedAffinity', 'docking_score', 'score', 'SCORE']:
                    if mol.HasProp(prop):
                        try:
                            scores.append(float(mol.GetProp(prop)))
                            break
                        except ValueError:
                            pass
        except ImportError:
            logger.warning("RDKit not available, cannot parse SDF scores")
        return scores


def dock_cluster_batch(
    cluster_dir: Path,
    ligand_library: Path,
    output_dir: Path,
    center: tuple,
    size: tuple = (20.0, 20.0, 20.0),
    exhaustiveness: int = 128,
    gpu_batch_size: int = 128,
    scoring: str = 'vina',
    num_modes: int = 10,
    n_clusters: Optional[int] = None
) -> Dict[str, List[DockingResult]]:
    """Dock a ligand library against all cluster representatives.
    
    This is a convenience function for batch docking against BioEmu clusters.
    
    Args:
        cluster_dir: Directory containing cluster_*.pdb files
        ligand_library: Path to ligand library (SDF, index file, or directory)
        output_dir: Base output directory
        center: (x, y, z) coordinates of docking box center
        size: (x, y, z) dimensions of docking box
        exhaustiveness: Search exhaustiveness
        gpu_batch_size: GPU batch size for Uni-Dock
        scoring: Scoring function ('vina', 'vinardo', 'ad4')
        num_modes: Number of poses per ligand
        n_clusters: Number of clusters to dock (None = all)
        
    Returns:
        Dictionary mapping cluster name to list of DockingResults
    """
    cluster_dir = Path(cluster_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find cluster files
    cluster_files = sorted(cluster_dir.glob('cluster_*.pdb'))
    if n_clusters:
        cluster_files = cluster_files[:n_clusters]
    
    logger.info(f"Docking against {len(cluster_files)} clusters")
    
    all_results = {}
    
    for cluster_pdb in cluster_files:
        cluster_name = cluster_pdb.stem
        cluster_output = output_dir / cluster_name
        
        logger.info(f"Processing {cluster_name}")
        
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
            exhaustiveness=exhaustiveness,
            gpu_batch_size=gpu_batch_size,
            scoring=scoring,
            num_modes=num_modes
        )
        
        engine = UniDockEngine(config)
        results = engine.dock()
        all_results[cluster_name] = results
        
        logger.info(f"Completed {cluster_name}: {len(results)} ligands docked")
    
    return all_results
