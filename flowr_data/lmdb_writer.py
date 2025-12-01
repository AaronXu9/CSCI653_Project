"""
LMDB writer for FLOWR.ROOT data serialization.

This module handles writing featurized protein-ligand systems to LMDB
(Lightning Memory-Mapped Database) format for high-performance I/O
during training.

Output format:
    - custom_data.lmdb: Main database with all featurized systems
    - data_statistics.npz: Marginal distributions for prior definition
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import numpy as np

from .featurizer import SystemFeatures, FeaturizationConfig

logger = logging.getLogger(__name__)


@dataclass
class DataStatistics:
    """Statistics about the dataset for prior distribution definition.
    
    These statistics are used by FLOWR.ROOT to define the prior
    distribution for the flow matching process.
    
    Attributes:
        n_systems: Total number of systems
        protein_element_counts: Count of each protein element type
        ligand_element_counts: Count of each ligand element type
        amino_acid_counts: Count of each amino acid type
        hybridization_counts: Count of each hybridization state
        ligand_atom_stats: Statistics for number of ligand atoms
        protein_atom_stats: Statistics for number of protein atoms
        affinity_stats: Statistics for affinity values
        position_stats: Statistics for atom positions
    """
    n_systems: int = 0
    protein_element_counts: np.ndarray = field(default_factory=lambda: np.zeros(7))
    ligand_element_counts: np.ndarray = field(default_factory=lambda: np.zeros(12))
    amino_acid_counts: np.ndarray = field(default_factory=lambda: np.zeros(21))
    hybridization_counts: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ligand_atom_stats: Dict[str, float] = field(default_factory=dict)
    protein_atom_stats: Dict[str, float] = field(default_factory=dict)
    affinity_stats: Dict[str, float] = field(default_factory=dict)
    position_stats: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def update(self, features: SystemFeatures):
        """Update statistics with a new system.
        
        Args:
            features: SystemFeatures object
        """
        self.n_systems += 1
        
        # Update element counts
        for elem_idx in features.protein_elements:
            if elem_idx < len(self.protein_element_counts):
                self.protein_element_counts[elem_idx] += 1
        
        for elem_idx in features.ligand_elements:
            if elem_idx < len(self.ligand_element_counts):
                self.ligand_element_counts[elem_idx] += 1
        
        # Update amino acid counts
        for aa_idx in features.protein_amino_acids:
            if aa_idx < len(self.amino_acid_counts):
                self.amino_acid_counts[aa_idx] += 1
        
        # Update hybridization counts
        for hyb_idx in features.ligand_hybridization:
            if hyb_idx < len(self.hybridization_counts):
                self.hybridization_counts[hyb_idx] += 1
    
    def finalize(self, 
                 all_ligand_atoms: List[int],
                 all_protein_atoms: List[int],
                 all_affinities: List[float],
                 all_positions: Optional[List[np.ndarray]] = None):
        """Compute final statistics.
        
        Args:
            all_ligand_atoms: List of ligand atom counts per system
            all_protein_atoms: List of protein atom counts per system
            all_affinities: List of affinity values
            all_positions: Optional list of all ligand positions
        """
        # Normalize counts to probabilities
        prot_total = self.protein_element_counts.sum()
        if prot_total > 0:
            self.protein_element_probs = self.protein_element_counts / prot_total
        
        lig_total = self.ligand_element_counts.sum()
        if lig_total > 0:
            self.ligand_element_probs = self.ligand_element_counts / lig_total
        
        aa_total = self.amino_acid_counts.sum()
        if aa_total > 0:
            self.amino_acid_probs = self.amino_acid_counts / aa_total
        
        hyb_total = self.hybridization_counts.sum()
        if hyb_total > 0:
            self.hybridization_probs = self.hybridization_counts / hyb_total
        
        # Compute atom count statistics
        if all_ligand_atoms:
            self.ligand_atom_stats = {
                'mean': float(np.mean(all_ligand_atoms)),
                'std': float(np.std(all_ligand_atoms)),
                'min': int(np.min(all_ligand_atoms)),
                'max': int(np.max(all_ligand_atoms)),
                'median': float(np.median(all_ligand_atoms))
            }
        
        if all_protein_atoms:
            self.protein_atom_stats = {
                'mean': float(np.mean(all_protein_atoms)),
                'std': float(np.std(all_protein_atoms)),
                'min': int(np.min(all_protein_atoms)),
                'max': int(np.max(all_protein_atoms)),
                'median': float(np.median(all_protein_atoms))
            }
        
        # Compute affinity statistics
        if all_affinities:
            self.affinity_stats = {
                'mean': float(np.mean(all_affinities)),
                'std': float(np.std(all_affinities)),
                'min': float(np.min(all_affinities)),
                'max': float(np.max(all_affinities)),
                'median': float(np.median(all_affinities))
            }
        
        # Compute position statistics
        if all_positions:
            all_pos = np.concatenate(all_positions, axis=0)
            self.position_stats = {
                'mean': all_pos.mean(axis=0),
                'std': all_pos.std(axis=0),
                'min': all_pos.min(axis=0),
                'max': all_pos.max(axis=0)
            }
    
    def save(self, path: Path):
        """Save statistics to NPZ file.
        
        Args:
            path: Output path for NPZ file
        """
        save_dict = {
            'n_systems': np.array([self.n_systems]),
            'protein_element_counts': self.protein_element_counts,
            'ligand_element_counts': self.ligand_element_counts,
            'amino_acid_counts': self.amino_acid_counts,
            'hybridization_counts': self.hybridization_counts,
        }
        
        # Add probability distributions if computed
        if hasattr(self, 'protein_element_probs'):
            save_dict['protein_element_probs'] = self.protein_element_probs
        if hasattr(self, 'ligand_element_probs'):
            save_dict['ligand_element_probs'] = self.ligand_element_probs
        if hasattr(self, 'amino_acid_probs'):
            save_dict['amino_acid_probs'] = self.amino_acid_probs
        if hasattr(self, 'hybridization_probs'):
            save_dict['hybridization_probs'] = self.hybridization_probs
        
        # Add scalar statistics as arrays
        for stat_name, stat_dict in [
            ('ligand_atom', self.ligand_atom_stats),
            ('protein_atom', self.protein_atom_stats),
            ('affinity', self.affinity_stats)
        ]:
            for key, value in stat_dict.items():
                save_dict[f'{stat_name}_{key}'] = np.array([value])
        
        # Add position statistics
        for key, value in self.position_stats.items():
            save_dict[f'position_{key}'] = value
        
        np.savez(str(path), **save_dict)
        logger.info(f"Saved statistics to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'DataStatistics':
        """Load statistics from NPZ file.
        
        Args:
            path: Path to NPZ file
            
        Returns:
            DataStatistics object
        """
        data = np.load(str(path), allow_pickle=True)
        
        stats = cls(
            n_systems=int(data['n_systems'][0]),
            protein_element_counts=data['protein_element_counts'],
            ligand_element_counts=data['ligand_element_counts'],
            amino_acid_counts=data['amino_acid_counts'],
            hybridization_counts=data['hybridization_counts'],
        )
        
        # Load probability distributions
        if 'protein_element_probs' in data:
            stats.protein_element_probs = data['protein_element_probs']
        if 'ligand_element_probs' in data:
            stats.ligand_element_probs = data['ligand_element_probs']
        if 'amino_acid_probs' in data:
            stats.amino_acid_probs = data['amino_acid_probs']
        if 'hybridization_probs' in data:
            stats.hybridization_probs = data['hybridization_probs']
        
        # Load scalar statistics
        for stat_name, stat_attr in [
            ('ligand_atom', 'ligand_atom_stats'),
            ('protein_atom', 'protein_atom_stats'),
            ('affinity', 'affinity_stats')
        ]:
            stat_dict = {}
            for key in ['mean', 'std', 'min', 'max', 'median']:
                full_key = f'{stat_name}_{key}'
                if full_key in data:
                    stat_dict[key] = float(data[full_key][0])
            setattr(stats, stat_attr, stat_dict)
        
        # Load position statistics
        position_stats = {}
        for key in ['mean', 'std', 'min', 'max']:
            full_key = f'position_{key}'
            if full_key in data:
                position_stats[key] = data[full_key]
        stats.position_stats = position_stats
        
        return stats


class LMDBWriter:
    """Writer for LMDB database format.
    
    Serializes featurized systems into LMDB for efficient I/O during
    FLOWR.ROOT training.
    
    Example:
        >>> writer = LMDBWriter(output_dir='flowr_data/final')
        >>> for features in feature_generator:
        ...     writer.add_system(features)
        >>> stats = writer.finalize()
    """
    
    def __init__(self,
                 output_dir: Path,
                 map_size: int = 1024 ** 4,  # 1TB default
                 db_name: str = 'custom_data'):
        """Initialize LMDB writer.
        
        Args:
            output_dir: Output directory
            map_size: Maximum database size in bytes
            db_name: Name for the database file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.output_dir / f'{db_name}.lmdb'
        self.stats_path = self.output_dir / 'data_statistics.npz'
        self.map_size = map_size
        
        self._env = None
        self._txn = None
        self._stats = DataStatistics()
        self._index = 0
        
        # Track values for final statistics
        self._ligand_atoms: List[int] = []
        self._protein_atoms: List[int] = []
        self._affinities: List[float] = []
        self._ligand_positions: List[np.ndarray] = []
    
    def _open_db(self):
        """Open LMDB database."""
        try:
            import lmdb
        except ImportError:
            raise ImportError("lmdb required: pip install lmdb")
        
        self._env = lmdb.open(
            str(self.db_path),
            map_size=self.map_size,
            subdir=False,
            readonly=False,
            meminit=False,
            map_async=True
        )
        self._txn = self._env.begin(write=True)
    
    def add_system(self, features: SystemFeatures, commit_every: int = 100):
        """Add a featurized system to the database.
        
        Args:
            features: SystemFeatures object
            commit_every: Commit transaction every N systems
        """
        if self._env is None:
            self._open_db()
        
        # Serialize features
        key = f'{self._index:08d}'.encode()
        value = pickle.dumps(features.to_dict())
        
        self._txn.put(key, value)
        
        # Update statistics
        self._stats.update(features)
        self._ligand_atoms.append(len(features.ligand_elements))
        self._protein_atoms.append(len(features.protein_elements))
        self._affinities.append(features.affinity)
        self._ligand_positions.append(features.ligand_pos)
        
        self._index += 1
        
        # Periodic commit
        if self._index % commit_every == 0:
            self._txn.commit()
            self._txn = self._env.begin(write=True)
            logger.info(f"Committed {self._index} systems")
    
    def add_systems(self, 
                    features_iter: Iterator[SystemFeatures],
                    total: Optional[int] = None):
        """Add multiple systems from an iterator.
        
        Args:
            features_iter: Iterator of SystemFeatures objects
            total: Total number (for progress bar)
        """
        try:
            from tqdm import tqdm
            features_iter = tqdm(features_iter, total=total, desc="Writing LMDB")
        except ImportError:
            pass
        
        for features in features_iter:
            self.add_system(features)
    
    def finalize(self) -> DataStatistics:
        """Finalize database and compute statistics.
        
        Returns:
            DataStatistics object
        """
        if self._txn is not None:
            # Store total count
            self._txn.put(b'__len__', str(self._index).encode())
            self._txn.commit()
        
        if self._env is not None:
            self._env.close()
        
        # Compute final statistics
        self._stats.finalize(
            self._ligand_atoms,
            self._protein_atoms,
            self._affinities,
            self._ligand_positions
        )
        
        # Save statistics
        self._stats.save(self.stats_path)
        
        logger.info(f"Finalized LMDB with {self._index} systems")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Statistics: {self.stats_path}")
        
        return self._stats
    
    def __enter__(self):
        self._open_db()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn is not None:
            self._txn.commit()
        if self._env is not None:
            self._env.close()


class LMDBReader:
    """Reader for LMDB database format.
    
    Example:
        >>> reader = LMDBReader('flowr_data/final/custom_data.lmdb')
        >>> for features in reader:
        ...     print(features.system_id)
        >>> reader.close()
    """
    
    def __init__(self, db_path: Path):
        """Initialize LMDB reader.
        
        Args:
            db_path: Path to LMDB file
        """
        try:
            import lmdb
        except ImportError:
            raise ImportError("lmdb required: pip install lmdb")
        
        self.db_path = Path(db_path)
        self._env = lmdb.open(
            str(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        
        with self._env.begin() as txn:
            length_bytes = txn.get(b'__len__')
            self._length = int(length_bytes.decode()) if length_bytes else 0
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> SystemFeatures:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")
        
        key = f'{idx:08d}'.encode()
        
        with self._env.begin() as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {key} not found")
            
            data = pickle.loads(value)
            return SystemFeatures.from_dict(data)
    
    def __iter__(self) -> Iterator[SystemFeatures]:
        for i in range(self._length):
            yield self[i]
    
    def close(self):
        """Close the database."""
        if self._env is not None:
            self._env.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def merge_lmdb_databases(db_paths: List[Path],
                         output_path: Path,
                         db_name: str = 'custom_data') -> DataStatistics:
    """Merge multiple LMDB databases into one.
    
    Args:
        db_paths: List of LMDB file paths
        output_path: Output directory
        db_name: Name for merged database
        
    Returns:
        DataStatistics for merged database
    """
    writer = LMDBWriter(output_path, db_name=db_name)
    
    for db_path in db_paths:
        logger.info(f"Merging {db_path}")
        reader = LMDBReader(db_path)
        
        for features in reader:
            writer.add_system(features)
        
        reader.close()
    
    return writer.finalize()
