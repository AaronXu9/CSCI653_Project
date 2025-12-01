"""
Base classes for docking engines.

This module defines the abstract interface that all docking engines must implement,
enabling easy extension to new docking software.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class DockingConfig:
    """Base configuration for docking engines.
    
    Attributes:
        receptor_path: Path to receptor file (PDB or PDBQT)
        ligand_paths: List of paths to ligand files or path to index file
        output_dir: Directory for output files
        center_x: X coordinate of docking box center
        center_y: Y coordinate of docking box center
        center_z: Z coordinate of docking box center
        size_x: X dimension of docking box (Angstroms)
        size_y: Y dimension of docking box (Angstroms)
        size_z: Z dimension of docking box (Angstroms)
        exhaustiveness: Search exhaustiveness (higher = more thorough)
        num_modes: Number of binding poses to generate per ligand
        energy_range: Maximum energy difference from best pose (kcal/mol)
        seed: Random seed for reproducibility
    """
    receptor_path: Path
    ligand_paths: Union[List[Path], Path]  # List of files or index file
    output_dir: Path
    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = 32
    num_modes: int = 9
    energy_range: float = 3.0
    seed: Optional[int] = None
    
    def __post_init__(self):
        self.receptor_path = Path(self.receptor_path)
        self.output_dir = Path(self.output_dir)
        if isinstance(self.ligand_paths, (str, Path)):
            self.ligand_paths = Path(self.ligand_paths)
        else:
            self.ligand_paths = [Path(p) for p in self.ligand_paths]


@dataclass
class DockingResult:
    """Container for docking results.
    
    Attributes:
        ligand_name: Name/identifier of the ligand
        receptor_name: Name/identifier of the receptor
        output_file: Path to output file with poses
        scores: List of docking scores for each pose
        best_score: Best (most negative) docking score
        num_poses: Number of poses generated
        metadata: Additional engine-specific information
    """
    ligand_name: str
    receptor_name: str
    output_file: Path
    scores: List[float] = field(default_factory=list)
    best_score: Optional[float] = None
    num_poses: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.scores and self.best_score is None:
            self.best_score = min(self.scores)
        self.num_poses = len(self.scores)


class DockingEngine(ABC):
    """Abstract base class for docking engines.
    
    All docking engines must implement this interface to be compatible
    with the docking pipeline.
    
    Example:
        >>> config = UniDockConfig(...)
        >>> engine = UniDockEngine(config)
        >>> results = engine.dock()
    """
    
    def __init__(self, config: DockingConfig):
        """Initialize docking engine with configuration.
        
        Args:
            config: Docking configuration object
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def check_installation(self) -> bool:
        """Check if the docking software is installed and accessible.
        
        Returns:
            True if software is available, False otherwise
        """
        pass
    
    @abstractmethod
    def prepare_receptor(self) -> Path:
        """Prepare receptor for docking (e.g., convert to PDBQT).
        
        Returns:
            Path to prepared receptor file
        """
        pass
    
    @abstractmethod
    def prepare_ligands(self) -> List[Path]:
        """Prepare ligands for docking.
        
        Returns:
            List of paths to prepared ligand files
        """
        pass
    
    @abstractmethod
    def dock(self) -> List[DockingResult]:
        """Execute the docking protocol.
        
        Returns:
            List of DockingResult objects
        """
        pass
    
    @abstractmethod
    def parse_output(self, output_path: Path) -> DockingResult:
        """Parse docking output file.
        
        Args:
            output_path: Path to output file
            
        Returns:
            Parsed DockingResult object
        """
        pass
    
    def get_engine_name(self) -> str:
        """Get the name of the docking engine.
        
        Returns:
            Engine name string
        """
        return self.__class__.__name__.replace('Engine', '')
    
    def get_version(self) -> Optional[str]:
        """Get the version of the docking software.
        
        Returns:
            Version string or None if not available
        """
        return None
