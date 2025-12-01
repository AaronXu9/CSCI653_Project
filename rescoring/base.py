"""
Base classes for rescoring engines.

This module defines the abstract interface for pose rescoring methods,
enabling extension to different scoring functions and ML models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PoseData:
    """Container for a single docking pose.
    
    Attributes:
        ligand_name: Identifier for the ligand
        receptor_name: Identifier for the receptor
        pose_file: Path to file containing the pose
        pose_index: Index of pose within multi-pose file
        original_score: Score from initial docking
        coordinates: Optional numpy array of coordinates
        metadata: Additional pose information
    """
    ligand_name: str
    receptor_name: str
    pose_file: Path
    pose_index: int = 0
    original_score: Optional[float] = None
    coordinates: Optional[Any] = None  # numpy array
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RescoringConfig:
    """Base configuration for rescoring engines.
    
    Attributes:
        receptor_path: Path to receptor file
        poses_path: Path to poses file or directory
        output_dir: Directory for output files
        batch_size: Number of poses to process in batch
        gpu_id: GPU device ID (-1 for CPU)
    """
    receptor_path: Path
    poses_path: Path  # Single file or directory
    output_dir: Path
    batch_size: int = 32
    gpu_id: int = 0
    
    def __post_init__(self):
        self.receptor_path = Path(self.receptor_path)
        self.poses_path = Path(self.poses_path)
        self.output_dir = Path(self.output_dir)


@dataclass 
class RescoringResult:
    """Container for rescoring results.
    
    Attributes:
        pose: The original PoseData object
        scores: Dictionary of score names to values
        primary_score: The main rescoring score
        confidence: Confidence/probability score if available
        ranking: Rank among poses for same ligand
        passed_filter: Whether pose passes quality thresholds
        metadata: Additional rescoring information
    """
    pose: PoseData
    scores: Dict[str, float] = field(default_factory=dict)
    primary_score: Optional[float] = None
    confidence: Optional[float] = None
    ranking: Optional[int] = None
    passed_filter: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def ligand_name(self) -> str:
        return self.pose.ligand_name
    
    @property
    def receptor_name(self) -> str:
        return self.pose.receptor_name


class RescoringEngine(ABC):
    """Abstract base class for rescoring engines.
    
    All rescoring methods must implement this interface for compatibility
    with the rescoring pipeline.
    
    Example:
        >>> config = GninaConfig(...)
        >>> rescorer = GninaRescorer(config)
        >>> results = rescorer.rescore(poses)
    """
    
    def __init__(self, config: RescoringConfig):
        """Initialize rescoring engine.
        
        Args:
            config: Rescoring configuration object
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
        """Check if the rescoring software is installed.
        
        Returns:
            True if software is available
        """
        pass
    
    @abstractmethod
    def rescore(self, poses: Optional[List[PoseData]] = None) -> List[RescoringResult]:
        """Rescore poses.
        
        Args:
            poses: Optional list of PoseData objects. If None, loads from config.
            
        Returns:
            List of RescoringResult objects
        """
        pass
    
    @abstractmethod
    def rescore_file(self, pose_file: Path) -> List[RescoringResult]:
        """Rescore all poses in a file.
        
        Args:
            pose_file: Path to file with poses (SDF, PDBQT, etc.)
            
        Returns:
            List of RescoringResult objects
        """
        pass
    
    def get_engine_name(self) -> str:
        """Get the name of the rescoring engine.
        
        Returns:
            Engine name string
        """
        return self.__class__.__name__.replace('Rescorer', '').replace('Engine', '')
    
    def get_score_names(self) -> List[str]:
        """Get names of scores computed by this engine.
        
        Returns:
            List of score names
        """
        return ['score']
    
    def rank_poses(self, results: List[RescoringResult], 
                   score_name: Optional[str] = None,
                   ascending: bool = True) -> List[RescoringResult]:
        """Rank poses by score.
        
        Args:
            results: List of RescoringResult objects
            score_name: Name of score to rank by (None = primary_score)
            ascending: If True, lower scores are better (like binding energy)
            
        Returns:
            Sorted list with ranking assigned
        """
        if score_name:
            key_func = lambda r: r.scores.get(score_name, float('inf'))
        else:
            key_func = lambda r: r.primary_score if r.primary_score is not None else float('inf')
        
        sorted_results = sorted(results, key=key_func, reverse=not ascending)
        
        for i, result in enumerate(sorted_results):
            result.ranking = i + 1
        
        return sorted_results
    
    def filter_by_score(self, results: List[RescoringResult],
                        threshold: float,
                        score_name: Optional[str] = None,
                        keep_above: bool = True) -> List[RescoringResult]:
        """Filter poses by score threshold.
        
        Args:
            results: List of RescoringResult objects
            threshold: Score threshold
            score_name: Score to filter by (None = primary_score)
            keep_above: If True, keep scores above threshold
            
        Returns:
            Filtered list of results
        """
        filtered = []
        for result in results:
            if score_name:
                score = result.scores.get(score_name)
            else:
                score = result.primary_score
            
            if score is None:
                continue
            
            if keep_above and score >= threshold:
                result.passed_filter = True
                filtered.append(result)
            elif not keep_above and score <= threshold:
                result.passed_filter = True
                filtered.append(result)
            else:
                result.passed_filter = False
        
        return filtered


class EnsembleRescorer:
    """Combine multiple rescoring methods.
    
    This class allows combining multiple rescoring engines to get
    consensus scores or multi-objective filtering.
    
    Example:
        >>> gnina = GninaRescorer(gnina_config)
        >>> rfscore = RFScoreRescorer(rf_config)  # hypothetical
        >>> ensemble = EnsembleRescorer([gnina, rfscore])
        >>> results = ensemble.rescore(poses)
    """
    
    def __init__(self, engines: List[RescoringEngine], 
                 weights: Optional[List[float]] = None):
        """Initialize ensemble rescorer.
        
        Args:
            engines: List of rescoring engines
            weights: Optional weights for combining scores
        """
        self.engines = engines
        self.weights = weights or [1.0] * len(engines)
        
        if len(self.weights) != len(self.engines):
            raise ValueError("Number of weights must match number of engines")
    
    def rescore(self, poses: List[PoseData]) -> List[RescoringResult]:
        """Rescore poses with all engines and combine.
        
        Args:
            poses: List of PoseData objects
            
        Returns:
            List of RescoringResult with combined scores
        """
        # Get results from each engine
        all_engine_results = []
        for engine in self.engines:
            results = engine.rescore(poses)
            all_engine_results.append(results)
        
        # Combine results
        combined_results = []
        for i, pose in enumerate(poses):
            scores = {}
            weighted_sum = 0.0
            
            for j, (engine, results) in enumerate(zip(self.engines, all_engine_results)):
                engine_name = engine.get_engine_name()
                
                # Find corresponding result
                result = next((r for r in results if r.pose.ligand_name == pose.ligand_name 
                              and r.pose.pose_index == pose.pose_index), None)
                
                if result:
                    for score_name, score_value in result.scores.items():
                        scores[f"{engine_name}_{score_name}"] = score_value
                    
                    if result.primary_score is not None:
                        weighted_sum += self.weights[j] * result.primary_score
            
            # Create combined result
            combined = RescoringResult(
                pose=pose,
                scores=scores,
                primary_score=weighted_sum / sum(self.weights),
                metadata={'engines': [e.get_engine_name() for e in self.engines]}
            )
            combined_results.append(combined)
        
        return combined_results
