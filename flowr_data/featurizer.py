"""
Featurization utilities for FLOWR.ROOT data preparation.

This module converts protein and ligand structures into the tensor format
required by FLOWR.ROOT's SE(3)-equivariant architecture.

Features extracted:
    Protein:
        - Atom positions (N_prot, 3)
        - Element types (one-hot encoded)
        - Amino acid types (one-hot encoded)
        - K-nearest neighbor graph edges
        
    Ligand:
        - Atom positions (N_lig, 3)
        - Element types (one-hot encoded)
        - Hybridization states
        - Formal charges
        - K-nearest neighbor graph edges
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

# Atom element vocabulary
PROTEIN_ELEMENTS = ['C', 'N', 'O', 'S', 'H', 'P', 'SE']  # Common protein atoms
LIGAND_ELEMENTS = [
    'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'H'
]

# Amino acid vocabulary (20 standard + X for unknown)
AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'UNK'
]

# Hybridization states for ligand atoms
HYBRIDIZATION_STATES = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']


@dataclass
class FeaturizationConfig:
    """Configuration for structure featurization.
    
    Attributes:
        knn_k: Number of nearest neighbors for graph construction
        max_protein_atoms: Maximum number of protein atoms (for padding)
        max_ligand_atoms: Maximum number of ligand atoms (for padding)
        pocket_radius: Radius (Å) around ligand to define pocket
        use_hydrogens: Whether to include hydrogen atoms
        center_on_ligand: Center coordinates on ligand centroid
        normalize_positions: Normalize positions to unit scale
    """
    knn_k: int = 16
    max_protein_atoms: int = 2000
    max_ligand_atoms: int = 100
    pocket_radius: float = 10.0
    use_hydrogens: bool = False
    center_on_ligand: bool = True
    normalize_positions: bool = False
    
    # Element vocabularies
    protein_elements: List[str] = field(default_factory=lambda: PROTEIN_ELEMENTS.copy())
    ligand_elements: List[str] = field(default_factory=lambda: LIGAND_ELEMENTS.copy())
    amino_acids: List[str] = field(default_factory=lambda: AMINO_ACIDS.copy())


@dataclass
class SystemFeatures:
    """Container for featurized protein-ligand system.
    
    This dataclass holds all features needed for FLOWR.ROOT training.
    
    Attributes:
        system_id: Unique identifier for this system
        protein_pos: (N_prot, 3) protein atom positions
        protein_elements: (N_prot,) element type indices
        protein_amino_acids: (N_prot,) amino acid type indices
        protein_edge_index: (2, E_prot) KNN edge indices
        ligand_pos: (N_lig, 3) ligand atom positions
        ligand_elements: (N_lig,) element type indices
        ligand_hybridization: (N_lig,) hybridization state indices
        ligand_formal_charge: (N_lig,) formal charges
        ligand_edge_index: (2, E_lig) KNN edge indices
        affinity: Binding affinity score (e.g., Gnina CNN affinity)
        metadata: Additional information
    """
    system_id: str
    protein_pos: np.ndarray
    protein_elements: np.ndarray
    protein_amino_acids: np.ndarray
    protein_edge_index: np.ndarray
    ligand_pos: np.ndarray
    ligand_elements: np.ndarray
    ligand_hybridization: np.ndarray
    ligand_formal_charge: np.ndarray
    ligand_edge_index: np.ndarray
    affinity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'system_id': self.system_id,
            'protein_pos': self.protein_pos.astype(np.float32),
            'protein_elements': self.protein_elements.astype(np.int32),
            'protein_amino_acids': self.protein_amino_acids.astype(np.int32),
            'protein_edge_index': self.protein_edge_index.astype(np.int64),
            'ligand_pos': self.ligand_pos.astype(np.float32),
            'ligand_elements': self.ligand_elements.astype(np.int32),
            'ligand_hybridization': self.ligand_hybridization.astype(np.int32),
            'ligand_formal_charge': self.ligand_formal_charge.astype(np.int32),
            'ligand_edge_index': self.ligand_edge_index.astype(np.int64),
            'affinity': np.float32(self.affinity),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemFeatures':
        """Create from dictionary."""
        return cls(
            system_id=data['system_id'],
            protein_pos=data['protein_pos'],
            protein_elements=data['protein_elements'],
            protein_amino_acids=data['protein_amino_acids'],
            protein_edge_index=data['protein_edge_index'],
            ligand_pos=data['ligand_pos'],
            ligand_elements=data['ligand_elements'],
            ligand_hybridization=data['ligand_hybridization'],
            ligand_formal_charge=data['ligand_formal_charge'],
            ligand_edge_index=data['ligand_edge_index'],
            affinity=float(data['affinity']),
            metadata=data.get('metadata', {})
        )


def build_knn_graph(positions: np.ndarray, k: int) -> np.ndarray:
    """Build K-nearest neighbor graph from positions.
    
    Args:
        positions: (N, 3) atom positions
        k: Number of nearest neighbors
        
    Returns:
        (2, E) edge index array where E = N * k
    """
    from scipy.spatial import cKDTree
    
    n_atoms = len(positions)
    if n_atoms <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    
    # Adjust k if fewer atoms than k
    k = min(k, n_atoms - 1)
    
    tree = cKDTree(positions)
    # Query k+1 because the first neighbor is the point itself
    _, indices = tree.query(positions, k=k + 1)
    
    # Build edge index
    src = np.repeat(np.arange(n_atoms), k)
    dst = indices[:, 1:].flatten()  # Exclude self-loops
    
    return np.stack([src, dst], axis=0)


class ProteinFeaturizer:
    """Featurizer for protein structures.
    
    Converts PDB files to tensor features for FLOWR.ROOT.
    
    Example:
        >>> config = FeaturizationConfig(pocket_radius=10.0)
        >>> featurizer = ProteinFeaturizer(config)
        >>> features = featurizer.featurize('receptor.pdb', ligand_center=[0, 0, 0])
    """
    
    def __init__(self, config: FeaturizationConfig):
        """Initialize protein featurizer.
        
        Args:
            config: Featurization configuration
        """
        self.config = config
        self._element_to_idx = {e: i for i, e in enumerate(config.protein_elements)}
        self._aa_to_idx = {aa: i for i, aa in enumerate(config.amino_acids)}
    
    def featurize(self,
                  pdb_path: Path,
                  ligand_center: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Featurize a protein structure.
        
        Args:
            pdb_path: Path to PDB file
            ligand_center: (3,) ligand centroid for pocket extraction
            
        Returns:
            Tuple of (positions, elements, amino_acids, edge_index)
        """
        try:
            from Bio.PDB import PDBParser
        except ImportError:
            raise ImportError("BioPython required: pip install biopython")
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_path))
        
        positions = []
        elements = []
        amino_acids = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip heteroatoms (water, ligands, etc.)
                    if residue.get_id()[0] != ' ':
                        continue
                    
                    res_name = residue.get_resname()
                    aa_idx = self._aa_to_idx.get(res_name, self._aa_to_idx.get('UNK', 0))
                    
                    for atom in residue:
                        # Skip hydrogens if configured
                        if not self.config.use_hydrogens and atom.element == 'H':
                            continue
                        
                        pos = atom.get_coord()
                        
                        # Apply pocket filter if ligand center provided
                        if ligand_center is not None:
                            dist = np.linalg.norm(pos - ligand_center)
                            if dist > self.config.pocket_radius:
                                continue
                        
                        positions.append(pos)
                        elem_idx = self._element_to_idx.get(
                            atom.element, 
                            self._element_to_idx.get('C', 0)  # Default to carbon
                        )
                        elements.append(elem_idx)
                        amino_acids.append(aa_idx)
        
        if not positions:
            logger.warning(f"No atoms found in {pdb_path}")
            positions = [[0, 0, 0]]
            elements = [0]
            amino_acids = [0]
        
        positions = np.array(positions, dtype=np.float32)
        elements = np.array(elements, dtype=np.int32)
        amino_acids = np.array(amino_acids, dtype=np.int32)
        
        # Build KNN graph
        edge_index = build_knn_graph(positions, self.config.knn_k)
        
        return positions, elements, amino_acids, edge_index
    
    def featurize_mdtraj(self,
                         pdb_path: Path,
                         ligand_center: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Featurize using MDTraj (faster for large proteins).
        
        Args:
            pdb_path: Path to PDB file
            ligand_center: (3,) ligand centroid for pocket extraction
            
        Returns:
            Tuple of (positions, elements, amino_acids, edge_index)
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError("MDTraj required: pip install mdtraj")
        
        traj = md.load(str(pdb_path))
        topology = traj.topology
        
        # Select atoms (exclude hydrogens if configured)
        if self.config.use_hydrogens:
            atom_indices = list(range(topology.n_atoms))
        else:
            atom_indices = topology.select('not element H')
        
        # Get positions (convert nm to Angstrom)
        positions = traj.xyz[0, atom_indices] * 10.0
        
        # Apply pocket filter
        if ligand_center is not None:
            distances = np.linalg.norm(positions - ligand_center, axis=1)
            pocket_mask = distances <= self.config.pocket_radius
            positions = positions[pocket_mask]
            atom_indices = [atom_indices[i] for i, m in enumerate(pocket_mask) if m]
        
        # Extract features
        elements = []
        amino_acids = []
        
        for idx in atom_indices:
            atom = topology.atom(idx)
            
            # Element
            elem_idx = self._element_to_idx.get(
                atom.element.symbol,
                self._element_to_idx.get('C', 0)
            )
            elements.append(elem_idx)
            
            # Amino acid
            if atom.residue.is_protein:
                aa_idx = self._aa_to_idx.get(
                    atom.residue.name,
                    self._aa_to_idx.get('UNK', 0)
                )
            else:
                aa_idx = self._aa_to_idx.get('UNK', 0)
            amino_acids.append(aa_idx)
        
        if len(positions) == 0:
            logger.warning(f"No atoms found in {pdb_path}")
            positions = np.zeros((1, 3), dtype=np.float32)
            elements = [0]
            amino_acids = [0]
        
        elements = np.array(elements, dtype=np.int32)
        amino_acids = np.array(amino_acids, dtype=np.int32)
        positions = positions.astype(np.float32)
        
        # Build KNN graph
        edge_index = build_knn_graph(positions, self.config.knn_k)
        
        return positions, elements, amino_acids, edge_index


class LigandFeaturizer:
    """Featurizer for ligand structures.
    
    Converts SDF files to tensor features for FLOWR.ROOT.
    
    Example:
        >>> config = FeaturizationConfig()
        >>> featurizer = LigandFeaturizer(config)
        >>> features = featurizer.featurize('ligand.sdf')
    """
    
    def __init__(self, config: FeaturizationConfig):
        """Initialize ligand featurizer.
        
        Args:
            config: Featurization configuration
        """
        self.config = config
        self._element_to_idx = {e: i for i, e in enumerate(config.ligand_elements)}
        self._hybrid_to_idx = {h: i for i, h in enumerate(HYBRIDIZATION_STATES)}
    
    def featurize(self,
                  sdf_path: Path,
                  mol_index: int = 0
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Featurize a ligand structure.
        
        Args:
            sdf_path: Path to SDF file
            mol_index: Index of molecule in multi-mol SDF
            
        Returns:
            Tuple of (positions, elements, hybridization, formal_charge, edge_index)
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError("RDKit required: pip install rdkit")
        
        # Load molecule
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=not self.config.use_hydrogens)
        
        mols = list(supplier)
        if mol_index >= len(mols) or mols[mol_index] is None:
            raise ValueError(f"Invalid molecule at index {mol_index} in {sdf_path}")
        
        mol = mols[mol_index]
        
        # Ensure we have 3D coordinates
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        
        conf = mol.GetConformer()
        
        positions = []
        elements = []
        hybridizations = []
        formal_charges = []
        
        for atom in mol.GetAtoms():
            # Position
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            
            # Element
            symbol = atom.GetSymbol()
            elem_idx = self._element_to_idx.get(
                symbol,
                self._element_to_idx.get('C', 0)
            )
            elements.append(elem_idx)
            
            # Hybridization
            hybrid = str(atom.GetHybridization())
            hybrid_idx = self._hybrid_to_idx.get(
                hybrid,
                self._hybrid_to_idx.get('UNSPECIFIED', 0)
            )
            hybridizations.append(hybrid_idx)
            
            # Formal charge
            formal_charges.append(atom.GetFormalCharge())
        
        positions = np.array(positions, dtype=np.float32)
        elements = np.array(elements, dtype=np.int32)
        hybridizations = np.array(hybridizations, dtype=np.int32)
        formal_charges = np.array(formal_charges, dtype=np.int32)
        
        # Build KNN graph
        edge_index = build_knn_graph(positions, self.config.knn_k)
        
        return positions, elements, hybridizations, formal_charges, edge_index
    
    def featurize_mol(self,
                      mol
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Featurize an RDKit molecule object directly.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Tuple of (positions, elements, hybridization, formal_charge, edge_index)
        """
        from rdkit.Chem import AllChem
        
        # Ensure we have 3D coordinates
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        
        conf = mol.GetConformer()
        
        positions = []
        elements = []
        hybridizations = []
        formal_charges = []
        
        for atom in mol.GetAtoms():
            # Position
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            
            # Element
            symbol = atom.GetSymbol()
            elem_idx = self._element_to_idx.get(
                symbol,
                self._element_to_idx.get('C', 0)
            )
            elements.append(elem_idx)
            
            # Hybridization
            hybrid = str(atom.GetHybridization())
            hybrid_idx = self._hybrid_to_idx.get(
                hybrid,
                self._hybrid_to_idx.get('UNSPECIFIED', 0)
            )
            hybridizations.append(hybrid_idx)
            
            # Formal charge
            formal_charges.append(atom.GetFormalCharge())
        
        positions = np.array(positions, dtype=np.float32)
        elements = np.array(elements, dtype=np.int32)
        hybridizations = np.array(hybridizations, dtype=np.int32)
        formal_charges = np.array(formal_charges, dtype=np.int32)
        
        # Build KNN graph
        edge_index = build_knn_graph(positions, self.config.knn_k)
        
        return positions, elements, hybridizations, formal_charges, edge_index


class SystemFeaturizer:
    """Combines protein and ligand featurization for complete systems.
    
    Example:
        >>> config = FeaturizationConfig()
        >>> featurizer = SystemFeaturizer(config)
        >>> features = featurizer.featurize_system(
        ...     'system_0001',
        ...     'protein.pdb',
        ...     'ligand.sdf',
        ...     affinity=-8.5
        ... )
    """
    
    def __init__(self, config: FeaturizationConfig):
        """Initialize system featurizer.
        
        Args:
            config: Featurization configuration
        """
        self.config = config
        self.protein_featurizer = ProteinFeaturizer(config)
        self.ligand_featurizer = LigandFeaturizer(config)
    
    def featurize_system(self,
                         system_id: str,
                         protein_path: Path,
                         ligand_path: Path,
                         affinity: float,
                         use_mdtraj: bool = False,
                         metadata: Optional[Dict[str, Any]] = None
                         ) -> SystemFeatures:
        """Featurize a complete protein-ligand system.
        
        Args:
            system_id: Unique identifier
            protein_path: Path to protein PDB
            ligand_path: Path to ligand SDF
            affinity: Binding affinity score
            use_mdtraj: Use MDTraj for protein (faster)
            metadata: Additional metadata
            
        Returns:
            SystemFeatures object
        """
        # Featurize ligand first to get center
        lig_pos, lig_elem, lig_hybrid, lig_charge, lig_edges = \
            self.ligand_featurizer.featurize(ligand_path)
        
        ligand_center = lig_pos.mean(axis=0)
        
        # Featurize protein (pocket extraction based on ligand center)
        if use_mdtraj:
            prot_pos, prot_elem, prot_aa, prot_edges = \
                self.protein_featurizer.featurize_mdtraj(protein_path, ligand_center)
        else:
            prot_pos, prot_elem, prot_aa, prot_edges = \
                self.protein_featurizer.featurize(protein_path, ligand_center)
        
        # Center coordinates on ligand if configured
        if self.config.center_on_ligand:
            prot_pos = prot_pos - ligand_center
            lig_pos = lig_pos - ligand_center
        
        return SystemFeatures(
            system_id=system_id,
            protein_pos=prot_pos,
            protein_elements=prot_elem,
            protein_amino_acids=prot_aa,
            protein_edge_index=prot_edges,
            ligand_pos=lig_pos,
            ligand_elements=lig_elem,
            ligand_hybridization=lig_hybrid,
            ligand_formal_charge=lig_charge,
            ligand_edge_index=lig_edges,
            affinity=affinity,
            metadata=metadata or {}
        )
