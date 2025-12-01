"""
Main data preparer for FLOWR.ROOT training data.

This module orchestrates the complete pipeline from filtered rescoring
results to LMDB-ready training data.

Pipeline:
    1. Export filtered poses to standardized directory structure
    2. Featurize protein and ligand structures
    3. Write features to LMDB database
    4. Compute and save data statistics
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import json

from .featurizer import (
    FeaturizationConfig, 
    SystemFeaturizer, 
    SystemFeatures
)
from .lmdb_writer import LMDBWriter, DataStatistics
from .exporter import PoseExporter, ExportConfig, ExportedSystem

logger = logging.getLogger(__name__)


@dataclass
class FlowrDataConfig:
    """Configuration for FLOWR data preparation.
    
    Attributes:
        output_dir: Base output directory for all data
        raw_dir: Subdirectory for raw exported poses
        processed_dir: Subdirectory for featurized pickles
        final_dir: Subdirectory for final LMDB
        
        Featurization:
        knn_k: K for KNN graph construction
        pocket_radius: Radius around ligand for pocket extraction
        use_hydrogens: Include hydrogen atoms
        center_on_ligand: Center coordinates on ligand centroid
        
        Export:
        copy_files: Copy files (vs symlink)
        overwrite: Overwrite existing data
        
        Processing:
        batch_size: Batch size for parallel processing
        n_workers: Number of parallel workers
        use_mdtraj: Use MDTraj for protein featurization
    """
    output_dir: Path
    raw_dir: Optional[Path] = None
    processed_dir: Optional[Path] = None
    final_dir: Optional[Path] = None
    
    # Featurization
    knn_k: int = 16
    pocket_radius: float = 10.0
    use_hydrogens: bool = False
    center_on_ligand: bool = True
    
    # Export
    copy_files: bool = True
    overwrite: bool = False
    
    # Processing
    batch_size: int = 32
    n_workers: int = 4
    use_mdtraj: bool = True
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        
        if self.raw_dir is None:
            self.raw_dir = self.output_dir / 'raw'
        else:
            self.raw_dir = Path(self.raw_dir)
        
        if self.processed_dir is None:
            self.processed_dir = self.output_dir / 'processed'
        else:
            self.processed_dir = Path(self.processed_dir)
        
        if self.final_dir is None:
            self.final_dir = self.output_dir / 'final'
        else:
            self.final_dir = Path(self.final_dir)
    
    def to_featurization_config(self) -> FeaturizationConfig:
        """Create FeaturizationConfig from this config."""
        return FeaturizationConfig(
            knn_k=self.knn_k,
            pocket_radius=self.pocket_radius,
            use_hydrogens=self.use_hydrogens,
            center_on_ligand=self.center_on_ligand
        )
    
    def to_export_config(self) -> ExportConfig:
        """Create ExportConfig from this config."""
        return ExportConfig(
            output_dir=self.raw_dir,
            copy_receptor=self.copy_files,
            copy_ligand=self.copy_files,
            overwrite=self.overwrite
        )


class FlowrDataPreparer:
    """Main class for preparing FLOWR.ROOT training data.
    
    This class orchestrates the complete pipeline from rescoring results
    to LMDB-ready training data.
    
    Example:
        >>> config = FlowrDataConfig(output_dir='flowr_training_data')
        >>> preparer = FlowrDataPreparer(config)
        >>> 
        >>> # From rescoring results
        >>> preparer.prepare_from_rescoring_results(results, cluster_pdbs)
        >>> 
        >>> # Or from existing raw directory
        >>> preparer.prepare_from_raw_directory()
        >>> 
        >>> # Generate final LMDB
        >>> stats = preparer.generate_lmdb()
    """
    
    def __init__(self, config: FlowrDataConfig):
        """Initialize data preparer.
        
        Args:
            config: FlowrDataConfig object
        """
        self.config = config
        self._setup_directories()
        
        self.featurizer = SystemFeaturizer(config.to_featurization_config())
        self.exporter = PoseExporter(config.to_export_config())
        
        self._exported_systems: List[ExportedSystem] = []
    
    def _setup_directories(self):
        """Create output directory structure."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config.final_dir.mkdir(parents=True, exist_ok=True)
    
    def export_from_rescoring_results(self,
                                       results: List[Any],
                                       cluster_pdb_map: Dict[str, Path]
                                       ) -> List[ExportedSystem]:
        """Export poses from rescoring results to raw directory.
        
        Args:
            results: List of RescoringResult objects
            cluster_pdb_map: Mapping of cluster names to PDB paths
            
        Returns:
            List of exported systems
        """
        self._exported_systems = self.exporter.export_from_rescoring_results(
            results, cluster_pdb_map
        )
        self.exporter.save_manifest(self._exported_systems)
        return self._exported_systems
    
    def export_from_directory(self,
                              docking_dir: Path,
                              cluster_dir: Path,
                              score_file: Optional[Path] = None,
                              score_threshold: float = 0.0
                              ) -> List[ExportedSystem]:
        """Export poses from docking output directory.
        
        Args:
            docking_dir: Directory with docking outputs
            cluster_dir: Directory with cluster PDBs
            score_file: Optional score file
            score_threshold: Minimum score threshold
            
        Returns:
            List of exported systems
        """
        self._exported_systems = self.exporter.export_from_directory(
            docking_dir=docking_dir,
            cluster_dir=cluster_dir,
            score_file=score_file,
            score_threshold=score_threshold
        )
        self.exporter.save_manifest(self._exported_systems)
        return self._exported_systems
    
    def load_exported_systems(self, 
                              manifest_path: Optional[Path] = None
                              ) -> List[ExportedSystem]:
        """Load previously exported systems from manifest.
        
        Args:
            manifest_path: Path to manifest (default: raw_dir/manifest.json)
            
        Returns:
            List of ExportedSystem objects
        """
        if manifest_path is None:
            manifest_path = self.config.raw_dir / 'manifest.json'
        
        self._exported_systems = PoseExporter.load_manifest(manifest_path)
        return self._exported_systems
    
    def featurize_system(self, system: ExportedSystem) -> Optional[SystemFeatures]:
        """Featurize a single exported system.
        
        Args:
            system: ExportedSystem object
            
        Returns:
            SystemFeatures if successful, None otherwise
        """
        try:
            features = self.featurizer.featurize_system(
                system_id=system.system_id,
                protein_path=system.protein_path,
                ligand_path=system.ligand_path,
                affinity=system.affinity,
                use_mdtraj=self.config.use_mdtraj,
                metadata={
                    'cluster_id': system.cluster_id,
                    'ligand_name': system.ligand_name,
                    **system.metadata
                }
            )
            return features
        except Exception as e:
            logger.error(f"Failed to featurize {system.system_id}: {e}")
            return None
    
    def featurize_all(self,
                      systems: Optional[List[ExportedSystem]] = None,
                      save_pickles: bool = False
                      ) -> Iterator[SystemFeatures]:
        """Featurize all systems.
        
        Args:
            systems: List of systems (default: loaded/exported systems)
            save_pickles: Save individual pickled features
            
        Yields:
            SystemFeatures objects
        """
        import pickle
        
        if systems is None:
            systems = self._exported_systems
        
        if not systems:
            # Try to load from manifest
            self.load_exported_systems()
            systems = self._exported_systems
        
        logger.info(f"Featurizing {len(systems)} systems...")
        
        for i, system in enumerate(systems):
            features = self.featurize_system(system)
            
            if features is not None:
                if save_pickles:
                    pickle_path = self.config.processed_dir / f'{system.system_id}.pkl'
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(features.to_dict(), f)
                
                yield features
            
            if (i + 1) % 100 == 0:
                logger.info(f"Featurized {i + 1}/{len(systems)} systems")
    
    def featurize_parallel(self,
                           systems: Optional[List[ExportedSystem]] = None,
                           n_workers: Optional[int] = None
                           ) -> Iterator[SystemFeatures]:
        """Featurize systems in parallel.
        
        Args:
            systems: List of systems
            n_workers: Number of workers (default: config.n_workers)
            
        Yields:
            SystemFeatures objects
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        if systems is None:
            systems = self._exported_systems
        
        if n_workers is None:
            n_workers = self.config.n_workers
        
        logger.info(f"Featurizing {len(systems)} systems with {n_workers} workers...")
        
        # Create a serializable featurization function
        feat_config = self.config.to_featurization_config()
        use_mdtraj = self.config.use_mdtraj
        
        def featurize_one(system_dict):
            """Worker function for parallel featurization."""
            from .featurizer import SystemFeaturizer
            featurizer = SystemFeaturizer(feat_config)
            
            try:
                features = featurizer.featurize_system(
                    system_id=system_dict['system_id'],
                    protein_path=Path(system_dict['protein_path']),
                    ligand_path=Path(system_dict['ligand_path']),
                    affinity=system_dict['affinity'],
                    use_mdtraj=use_mdtraj,
                    metadata=system_dict.get('metadata', {})
                )
                return features.to_dict()
            except Exception as e:
                return None
        
        # Convert systems to dicts for serialization
        system_dicts = [s.to_dict() for s in systems]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(featurize_one, sd): sd 
                       for sd in system_dicts}
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    yield SystemFeatures.from_dict(result)
    
    def generate_lmdb(self,
                      systems: Optional[List[ExportedSystem]] = None,
                      parallel: bool = False,
                      db_name: str = 'custom_data'
                      ) -> DataStatistics:
        """Generate LMDB database from systems.
        
        Args:
            systems: List of systems (default: loaded/exported)
            parallel: Use parallel featurization
            db_name: Name for database file
            
        Returns:
            DataStatistics object
        """
        writer = LMDBWriter(
            output_dir=self.config.final_dir,
            db_name=db_name
        )
        
        if parallel:
            features_iter = self.featurize_parallel(systems)
        else:
            features_iter = self.featurize_all(systems)
        
        for features in features_iter:
            writer.add_system(features)
        
        stats = writer.finalize()
        
        logger.info(f"Generated LMDB at {self.config.final_dir / f'{db_name}.lmdb'}")
        logger.info(f"Total systems: {stats.n_systems}")
        
        return stats
    
    def prepare_from_rescoring_results(self,
                                        results: List[Any],
                                        cluster_pdb_map: Dict[str, Path],
                                        parallel: bool = False
                                        ) -> DataStatistics:
        """Complete pipeline from rescoring results to LMDB.
        
        Args:
            results: List of RescoringResult objects
            cluster_pdb_map: Mapping of cluster names to PDB paths
            parallel: Use parallel processing
            
        Returns:
            DataStatistics object
        """
        # Export
        logger.info("Step 1: Exporting poses to raw directory...")
        exported = self.export_from_rescoring_results(results, cluster_pdb_map)
        
        # Generate LMDB
        logger.info("Step 2: Featurizing and generating LMDB...")
        stats = self.generate_lmdb(exported, parallel=parallel)
        
        logger.info("Data preparation complete!")
        return stats
    
    def prepare_from_raw_directory(self,
                                    raw_dir: Optional[Path] = None,
                                    parallel: bool = False
                                    ) -> DataStatistics:
        """Generate LMDB from existing raw directory.
        
        Args:
            raw_dir: Raw directory with exported systems
            parallel: Use parallel processing
            
        Returns:
            DataStatistics object
        """
        if raw_dir is not None:
            manifest_path = raw_dir / 'manifest.json'
        else:
            manifest_path = self.config.raw_dir / 'manifest.json'
        
        logger.info("Loading exported systems from manifest...")
        systems = self.load_exported_systems(manifest_path)
        
        logger.info(f"Loaded {len(systems)} systems")
        
        # Generate LMDB
        logger.info("Featurizing and generating LMDB...")
        stats = self.generate_lmdb(systems, parallel=parallel)
        
        logger.info("Data preparation complete!")
        return stats
    
    def validate_lmdb(self, 
                      db_path: Optional[Path] = None,
                      n_samples: int = 10
                      ) -> bool:
        """Validate generated LMDB database.
        
        Args:
            db_path: Path to LMDB (default: final_dir/custom_data.lmdb)
            n_samples: Number of samples to validate
            
        Returns:
            True if validation passes
        """
        from .lmdb_writer import LMDBReader
        
        if db_path is None:
            db_path = self.config.final_dir / 'custom_data.lmdb'
        
        logger.info(f"Validating {db_path}...")
        
        try:
            reader = LMDBReader(db_path)
            logger.info(f"Database contains {len(reader)} systems")
            
            # Check random samples
            import random
            indices = random.sample(range(len(reader)), min(n_samples, len(reader)))
            
            for idx in indices:
                features = reader[idx]
                
                # Validate structure
                assert features.protein_pos.shape[1] == 3
                assert features.ligand_pos.shape[1] == 3
                assert len(features.protein_elements) == len(features.protein_pos)
                assert len(features.ligand_elements) == len(features.ligand_pos)
                
                logger.debug(f"Sample {idx}: protein={len(features.protein_pos)}, "
                            f"ligand={len(features.ligand_pos)}, "
                            f"affinity={features.affinity:.2f}")
            
            reader.close()
            logger.info("Validation passed!")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
