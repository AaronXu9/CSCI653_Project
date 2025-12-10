"""
Utility functions for molecular docking preparation.

This module provides helper functions for preparing receptors and ligands
for docking, including file format conversion and binding site detection.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def run_command(cmd: List[str], check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with error handling.
    
    Args:
        cmd: Command and arguments as list
        check: Whether to raise exception on non-zero return code
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        CompletedProcess instance
        
    Raises:
        subprocess.CalledProcessError: If command fails and check=True
    """
    logger.debug(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=capture_output, text=True)
    if result.returncode != 0 and not check:
        logger.warning(f"Command returned non-zero: {result.stderr}")
    return result


def prepare_receptor_pdbqt(
    pdb_path: Path,
    output_path: Optional[Path] = None,
    add_hydrogens: bool = True,
    remove_water: bool = True,
    remove_heteroatoms: bool = False,
    ph: float = 7.4
) -> Path:
    """Convert PDB receptor to PDBQT format using Open Babel or ADFRsuite.
    
    Args:
        pdb_path: Path to input PDB file
        output_path: Path for output PDBQT file (default: same name with .pdbqt)
        add_hydrogens: Whether to add hydrogens
        remove_water: Whether to remove water molecules
        remove_heteroatoms: Whether to remove heteroatoms (ligands, etc.)
        ph: pH for hydrogen addition
        
    Returns:
        Path to output PDBQT file
        
    Raises:
        RuntimeError: If conversion fails
    """
    pdb_path = Path(pdb_path)
    if output_path is None:
        output_path = pdb_path.with_suffix('.pdbqt')
    else:
        output_path = Path(output_path)
    
    # Try using prepare_receptor from ADFRsuite first (preferred)
    try:
        cmd = ['prepare_receptor', '-r', str(pdb_path), '-o', str(output_path)]
        if not add_hydrogens:
            cmd.append('-A')
            cmd.append('None')
        result = run_command(cmd, check=False)
        if result.returncode == 0 and output_path.exists():
            logger.info(f"Prepared receptor using ADFRsuite: {output_path}")
            return output_path
    except FileNotFoundError:
        logger.debug("ADFRsuite prepare_receptor not found, trying Open Babel")
    
    # Fallback to Open Babel
    try:
        # Build Open Babel command
        cmd = ['obabel', str(pdb_path), '-O', str(output_path)]
        
        if add_hydrogens:
            cmd.extend(['-h', '-p', str(ph)])
        
        if remove_water:
            cmd.extend(['--delete', 'HOH'])
        
        result = run_command(cmd, check=True)
        
        # Post-process to remove ROOT/ENDROOT/TORSDOF if present (Open Babel adds them sometimes)
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        with open(output_path, 'w') as f:
            for line in lines:
                if line.startswith('ROOT') or line.startswith('ENDROOT') or line.startswith('TORSDOF'):
                    continue
                f.write(line)

        logger.info(f"Prepared receptor using Open Babel: {output_path}")
        return output_path
        
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            f"Failed to convert receptor to PDBQT. "
            f"Please install ADFRsuite or Open Babel. Error: {e}"
        )


def prepare_ligand_pdbqt(
    ligand_path: Path,
    output_path: Optional[Path] = None,
    add_hydrogens: bool = True,
    gen_3d: bool = False,
    ph: float = 7.4
) -> Path:
    """Convert ligand to PDBQT format using Open Babel or ADFRsuite.
    
    Args:
        ligand_path: Path to input ligand file (SDF, MOL2, PDB, SMILES, etc.)
        output_path: Path for output PDBQT file
        add_hydrogens: Whether to add hydrogens
        gen_3d: Whether to generate 3D coordinates (for 2D inputs like SMILES)
        ph: pH for protonation state
        
    Returns:
        Path to output PDBQT file
    """
    ligand_path = Path(ligand_path)
    if output_path is None:
        output_path = ligand_path.with_suffix('.pdbqt')
    else:
        output_path = Path(output_path)
    
    # Try ADFRsuite prepare_ligand first
    try:
        cmd = ['prepare_ligand', '-l', str(ligand_path), '-o', str(output_path)]
        result = run_command(cmd, check=False)
        if result.returncode == 0 and output_path.exists():
            logger.info(f"Prepared ligand using ADFRsuite: {output_path}")
            return output_path
    except FileNotFoundError:
        logger.debug("ADFRsuite prepare_ligand not found, trying Open Babel")
    
    # Fallback to Open Babel
    try:
        cmd = ['obabel', str(ligand_path), '-O', str(output_path)]
        
        if add_hydrogens:
            cmd.extend(['-h', '-p', str(ph)])
        
        if gen_3d:
            cmd.append('--gen3d')
        
        result = run_command(cmd, check=True)
        logger.info(f"Prepared ligand using Open Babel: {output_path}")
        return output_path
        
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            f"Failed to convert ligand to PDBQT. "
            f"Please install ADFRsuite or Open Babel. Error: {e}"
        )


def batch_prepare_ligands(
    ligand_paths: List[Path],
    output_dir: Path,
    add_hydrogens: bool = True,
    n_jobs: int = 1
) -> List[Path]:
    """Batch prepare multiple ligands for docking.
    
    Args:
        ligand_paths: List of input ligand files
        output_dir: Directory for output files
        add_hydrogens: Whether to add hydrogens
        n_jobs: Number of parallel jobs (future implementation)
        
    Returns:
        List of output PDBQT paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prepared_ligands = []
    for ligand_path in ligand_paths:
        try:
            output_path = output_dir / f"{ligand_path.stem}.pdbqt"
            prepared = prepare_ligand_pdbqt(ligand_path, output_path, add_hydrogens)
            prepared_ligands.append(prepared)
        except Exception as e:
            logger.warning(f"Failed to prepare ligand {ligand_path}: {e}")
    
    return prepared_ligands


def detect_binding_site(
    pdb_path: Path,
    method: str = 'geometric_center',
    residue_ids: Optional[List[int]] = None,
    chain: Optional[str] = None,
    ligand_name: Optional[str] = None,
    padding: float = 5.0
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Detect binding site center and dimensions.
    
    Args:
        pdb_path: Path to PDB file
        method: Detection method - 'geometric_center', 'residues', 'ligand'
        residue_ids: List of residue IDs defining the binding site (for 'residues' method)
        chain: Chain ID to use
        ligand_name: Ligand residue name (for 'ligand' method)
        padding: Extra space around detected site (Angstroms)
        
    Returns:
        Tuple of (center, size) where each is (x, y, z)
    """
    try:
        import mdtraj as md
    except ImportError:
        raise ImportError("MDTraj is required for binding site detection")
    
    traj = md.load(str(pdb_path))
    topology = traj.topology
    
    if method == 'residues' and residue_ids is not None:
        # Select atoms based on residue IDs
        selection = ' or '.join([f'resid {r}' for r in residue_ids])
        if chain:
            selection = f'({selection}) and chainid {chain}'
        atom_indices = topology.select(selection)
        
    elif method == 'ligand' and ligand_name is not None:
        # Select ligand atoms
        atom_indices = topology.select(f'resname {ligand_name}')
        
    elif method == 'geometric_center':
        # Use CA atoms of all residues
        atom_indices = topology.select('name CA')
        
    else:
        raise ValueError(f"Invalid method or missing parameters: {method}")
    
    if len(atom_indices) == 0:
        raise ValueError("No atoms found for binding site detection")
    
    # Get coordinates (convert nm to Angstroms)
    coords = traj.xyz[0, atom_indices, :] * 10.0
    
    # Calculate center and size
    center = coords.mean(axis=0)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    size = (max_coords - min_coords) + 2 * padding
    
    return tuple(center.tolist()), tuple(size.tolist())


def create_ligand_index_file(
    ligand_paths: List[Path],
    output_path: Path
) -> Path:
    """Create an index file listing ligand paths for batch docking.
    
    Args:
        ligand_paths: List of ligand file paths
        output_path: Path for output index file
        
    Returns:
        Path to created index file
    """
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        for path in ligand_paths:
            f.write(f"{path}\n")
    logger.info(f"Created ligand index file: {output_path}")
    return output_path


def parse_sdf_file(sdf_path: Path) -> List[Dict[str, Any]]:
    """Parse an SDF file and extract molecule information.
    
    Args:
        sdf_path: Path to SDF file
        
    Returns:
        List of dictionaries with molecule data
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit is required for SDF parsing")
    
    molecules = []
    suppl = Chem.SDMolSupplier(str(sdf_path))
    
    for mol in suppl:
        if mol is None:
            continue
        
        mol_data = {
            'name': mol.GetProp('_Name') if mol.HasProp('_Name') else 'Unknown',
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'properties': {}
        }
        
        # Extract all properties
        for prop in mol.GetPropsAsDict():
            mol_data['properties'][prop] = mol.GetProp(prop)
        
        molecules.append(mol_data)
    
    return molecules


def get_box_from_reference_ligand(
    ligand_path: Path,
    padding: float = 5.0
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Calculate docking box from a reference ligand.
    
    Args:
        ligand_path: Path to reference ligand file
        padding: Extra space around ligand (Angstroms)
        
    Returns:
        Tuple of (center, size)
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit is required for ligand box calculation")
    
    # Load molecule based on file type
    ligand_path = Path(ligand_path)
    suffix = ligand_path.suffix.lower()
    
    if suffix == '.sdf':
        mol = Chem.SDMolSupplier(str(ligand_path))[0]
    elif suffix == '.mol2':
        mol = Chem.MolFromMol2File(str(ligand_path))
    elif suffix in ['.pdb', '.pdbqt']:
        mol = Chem.MolFromPDBFile(str(ligand_path))
    else:
        raise ValueError(f"Unsupported ligand format: {suffix}")
    
    if mol is None:
        raise ValueError(f"Failed to load ligand: {ligand_path}")
    
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    
    center = coords.mean(axis=0)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    size = (max_coords - min_coords) + 2 * padding
    
    return tuple(center.tolist()), tuple(size.tolist())
