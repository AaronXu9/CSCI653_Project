"""
Pose filtering utilities for rescoring pipelines.

This module provides flexible filtering and aggregation utilities
for processing rescoring results across multiple scoring methods.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Union
import logging
import json
import csv

from .base import RescoringResult, PoseData

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for score-based filtering.
    
    Attributes:
        score_name: Name of score to filter by
        threshold: Threshold value
        keep_above: If True, keep values above threshold
        required: If True, poses without this score are rejected
    """
    score_name: str
    threshold: float
    keep_above: bool = True
    required: bool = True


@dataclass
class FilterResult:
    """Result of filtering operation.
    
    Attributes:
        passed: List of results that passed all filters
        failed: List of results that failed filters
        stats: Statistics about filtering
    """
    passed: List[RescoringResult] = field(default_factory=list)
    failed: List[RescoringResult] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        total = len(self.passed) + len(self.failed)
        return len(self.passed) / total if total > 0 else 0.0
    
    def summary(self) -> str:
        return (f"Passed: {len(self.passed)}, "
                f"Failed: {len(self.failed)}, "
                f"Rate: {self.pass_rate:.1%}")


class ScoreFilter:
    """Flexible multi-criteria pose filter.
    
    Allows combining multiple score thresholds with AND/OR logic
    and custom filtering functions.
    
    Example:
        >>> filter = ScoreFilter()
        >>> filter.add_threshold('CNNscore', 0.9, keep_above=True)
        >>> filter.add_threshold('CNNaffinity', -7.0, keep_above=False)
        >>> result = filter.apply(rescoring_results)
    """
    
    def __init__(self, logic: str = 'AND'):
        """Initialize filter.
        
        Args:
            logic: How to combine filters - 'AND' or 'OR'
        """
        self.logic = logic.upper()
        self.filters: List[FilterConfig] = []
        self.custom_filters: List[Callable[[RescoringResult], bool]] = []
    
    def add_threshold(self, 
                      score_name: str,
                      threshold: float,
                      keep_above: bool = True,
                      required: bool = True) -> 'ScoreFilter':
        """Add a score threshold filter.
        
        Args:
            score_name: Name of score to filter by
            threshold: Threshold value
            keep_above: If True, keep scores above threshold
            required: If True, reject poses missing this score
            
        Returns:
            Self for chaining
        """
        self.filters.append(FilterConfig(
            score_name=score_name,
            threshold=threshold,
            keep_above=keep_above,
            required=required
        ))
        return self
    
    def add_custom_filter(self, 
                          func: Callable[[RescoringResult], bool],
                          name: Optional[str] = None) -> 'ScoreFilter':
        """Add a custom filter function.
        
        Args:
            func: Function taking RescoringResult, returning True to keep
            name: Optional name for the filter
            
        Returns:
            Self for chaining
        """
        self.custom_filters.append(func)
        return self
    
    def _check_threshold(self, result: RescoringResult, 
                         config: FilterConfig) -> Optional[bool]:
        """Check if result passes a threshold filter.
        
        Args:
            result: RescoringResult to check
            config: FilterConfig with threshold
            
        Returns:
            True if passes, False if fails, None if score not present
        """
        score = result.scores.get(config.score_name)
        
        if score is None:
            if config.score_name == 'primary' and result.primary_score is not None:
                score = result.primary_score
            else:
                return None  # Score not present
        
        if config.keep_above:
            return score >= config.threshold
        else:
            return score <= config.threshold
    
    def check(self, result: RescoringResult) -> bool:
        """Check if a single result passes all filters.
        
        Args:
            result: RescoringResult to check
            
        Returns:
            True if result passes filters
        """
        threshold_results = []
        
        for config in self.filters:
            check = self._check_threshold(result, config)
            if check is None:
                if config.required:
                    return False  # Required score missing
                continue
            threshold_results.append(check)
        
        # Apply custom filters
        custom_results = [f(result) for f in self.custom_filters]
        
        all_results = threshold_results + custom_results
        
        if not all_results:
            return True  # No filters defined
        
        if self.logic == 'AND':
            return all(all_results)
        else:  # OR
            return any(all_results)
    
    def apply(self, results: List[RescoringResult]) -> FilterResult:
        """Apply filters to a list of results.
        
        Args:
            results: List of RescoringResult objects
            
        Returns:
            FilterResult with passed and failed lists
        """
        passed = []
        failed = []
        
        for result in results:
            if self.check(result):
                result.passed_filter = True
                passed.append(result)
            else:
                result.passed_filter = False
                failed.append(result)
        
        # Compute statistics
        stats = {
            'total': len(results),
            'passed': len(passed),
            'failed': len(failed),
            'pass_rate': len(passed) / len(results) if results else 0.0,
            'filters_applied': len(self.filters) + len(self.custom_filters)
        }
        
        return FilterResult(passed=passed, failed=failed, stats=stats)


class ResultAggregator:
    """Aggregate rescoring results across multiple clusters/receptors.
    
    Provides utilities for combining, ranking, and exporting results
    from ensemble docking campaigns.
    """
    
    def __init__(self):
        """Initialize aggregator."""
        self.results: Dict[str, List[RescoringResult]] = {}
    
    def add_results(self, cluster_name: str, 
                    results: List[RescoringResult]) -> None:
        """Add results for a cluster.
        
        Args:
            cluster_name: Name of the cluster/receptor
            results: List of RescoringResult objects
        """
        self.results[cluster_name] = results
    
    def get_all_results(self) -> List[RescoringResult]:
        """Get all results as a flat list.
        
        Returns:
            List of all RescoringResult objects
        """
        all_results = []
        for cluster_results in self.results.values():
            all_results.extend(cluster_results)
        return all_results
    
    def get_unique_ligands(self) -> Dict[str, List[RescoringResult]]:
        """Group results by ligand name.
        
        Returns:
            Dictionary mapping ligand name to list of results
        """
        by_ligand = {}
        for result in self.get_all_results():
            ligand = result.ligand_name
            if ligand not in by_ligand:
                by_ligand[ligand] = []
            by_ligand[ligand].append(result)
        return by_ligand
    
    def get_best_per_ligand(self, 
                            score_name: str = 'CNNscore',
                            higher_is_better: bool = True) -> List[RescoringResult]:
        """Get best-scoring result for each ligand.
        
        Args:
            score_name: Score to rank by
            higher_is_better: If True, higher score is better
            
        Returns:
            List with one result per ligand
        """
        by_ligand = self.get_unique_ligands()
        best_results = []
        
        for ligand, results in by_ligand.items():
            # Sort by score
            sorted_results = sorted(
                results,
                key=lambda r: r.scores.get(score_name, float('-inf') if higher_is_better else float('inf')),
                reverse=higher_is_better
            )
            if sorted_results:
                best_results.append(sorted_results[0])
        
        return best_results
    
    def get_consensus_binders(self,
                              min_clusters: int = 3,
                              score_name: str = 'CNNscore',
                              threshold: float = 0.9) -> List[str]:
        """Find ligands that bind well to multiple clusters.
        
        Args:
            min_clusters: Minimum number of clusters with good binding
            score_name: Score to evaluate
            threshold: Score threshold for "good" binding
            
        Returns:
            List of ligand names with consensus binding
        """
        by_ligand = self.get_unique_ligands()
        consensus = []
        
        for ligand, results in by_ligand.items():
            good_clusters = sum(
                1 for r in results 
                if r.scores.get(score_name, 0) >= threshold
            )
            if good_clusters >= min_clusters:
                consensus.append(ligand)
        
        return consensus
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame.
        
        Returns:
            DataFrame with all results
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for DataFrame export")
        
        rows = []
        for cluster, results in self.results.items():
            for result in results:
                row = {
                    'cluster': cluster,
                    'ligand': result.ligand_name,
                    'pose_index': result.pose.pose_index,
                    'passed_filter': result.passed_filter,
                    **result.scores
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def to_csv(self, output_path: Path) -> None:
        """Export results to CSV file.
        
        Args:
            output_path: Path for output CSV
        """
        all_results = self.get_all_results()
        if not all_results:
            logger.warning("No results to export")
            return
        
        # Collect all score names
        all_scores = set()
        for result in all_results:
            all_scores.update(result.scores.keys())
        
        fieldnames = ['cluster', 'ligand', 'pose_index', 'passed_filter'] + sorted(all_scores)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for cluster, results in self.results.items():
                for result in results:
                    row = {
                        'cluster': cluster,
                        'ligand': result.ligand_name,
                        'pose_index': result.pose.pose_index,
                        'passed_filter': result.passed_filter,
                        **result.scores
                    }
                    writer.writerow(row)
        
        logger.info(f"Exported {len(all_results)} results to {output_path}")
    
    def to_json(self, output_path: Path) -> None:
        """Export results to JSON file.
        
        Args:
            output_path: Path for output JSON
        """
        data = {}
        for cluster, results in self.results.items():
            data[cluster] = [
                {
                    'ligand': r.ligand_name,
                    'pose_index': r.pose.pose_index,
                    'scores': r.scores,
                    'passed_filter': r.passed_filter,
                    'pose_file': str(r.pose.pose_file)
                }
                for r in results
            ]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported results to {output_path}")
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        all_results = self.get_all_results()
        unique_ligands = self.get_unique_ligands()
        
        return {
            'n_clusters': len(self.results),
            'n_total_results': len(all_results),
            'n_unique_ligands': len(unique_ligands),
            'n_passed': sum(1 for r in all_results if r.passed_filter),
            'per_cluster': {
                cluster: {
                    'total': len(results),
                    'passed': sum(1 for r in results if r.passed_filter)
                }
                for cluster, results in self.results.items()
            }
        }
