"""
Pose exporter for FLOWR.ROOT directory structure.

This module exports filtered docking poses to the standardized directory
structure expected by FLOWR.ROOT preprocessing scripts:

    /project/training_data/raw/
    ├── system_0001/
    │   ├── system_0001_ligand.sdf
    │   └── system_0001_protein.pdb
    ├── system_0002/
    │   ├── system_0002_ligand.sdf
    │   └── system_0002_protein.pdb
    ...

Each "system" corresponds to a successful docking result (a specific 
BioEmu cluster + a specific high-scoring ligand).
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for pose export.
    
    Attributes:
        output_dir: Base output directory
        system_prefix: Prefix for system directories (e.g., 'system')
        copy_receptor: Whether to copy receptor files (vs symlink)
        copy_ligand: Whether to copy ligand files (vs symlink)
        overwrite: Whether to overwrite existing systems
    """
    output_dir: Path
    system_prefix: str = 'system'
    copy_receptor: bool = True
    copy_ligand: bool = True
    overwrite: bool = False
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class ExportedSystem:
    """Information about an exported system.
    
    Attributes:
        system_id: Unique system identifier
        system_dir: Path to system directory
        protein_path: Path to protein PDB file
        ligand_path: Path to ligand SDF file
        affinity: Binding affinity score
        cluster_id: Source cluster identifier
        ligand_name: Original ligand name
        pose_index: Pose index in original docking output
        metadata: Additional information
    """
    system_id: str
    system_dir: Path
    protein_path: Path
    ligand_path: Path
    affinity: float
    cluster_id: Optional[str] = None
    ligand_name: Optional[str] = None
    pose_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'system_id': self.system_id,
            'system_dir': str(self.system_dir),
            'protein_path': str(self.protein_path),
            'ligand_path': str(self.ligand_path),
            'affinity': self.affinity,
            'cluster_id': self.cluster_id,
            'ligand_name': self.ligand_name,
            'pose_index': self.pose_index,
            'metadata': self.metadata
        }


class PoseExporter:
    """Exports docking poses to FLOWR.ROOT directory structure.
    
    This class converts rescoring results into the standardized format
    expected by FLOWR.ROOT's preprocessing scripts.
    
    Example:
        >>> config = ExportConfig(output_dir='training_data/raw')
        >>> exporter = PoseExporter(config)
        >>> exported = exporter.export_from_rescoring(results, cluster_pdbs)
        >>> exporter.save_manifest(exported)
    """
    
    def __init__(self, config: ExportConfig):
        """Initialize pose exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._system_counter = 0
    
    def _get_next_system_id(self) -> str:
        """Get the next system ID."""
        self._system_counter += 1
        return f"{self.config.system_prefix}_{self._system_counter:04d}"
    
    def _extract_pose_to_sdf(self,
                             source_path: Path,
                             dest_path: Path,
                             pose_index: int = 0) -> bool:
        """Extract a single pose from multi-pose file to SDF.
        
        Args:
            source_path: Path to source file (SDF/PDBQT)
            dest_path: Destination SDF path
            pose_index: Index of pose to extract
            
        Returns:
            True if successful
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise ImportError("RDKit required: pip install rdkit")
        
        suffix = source_path.suffix.lower()
        
        if suffix == '.sdf':
            # Read SDF and extract specific molecule
            supplier = Chem.SDMolSupplier(str(source_path), removeHs=False)
            mols = list(supplier)
            
            if pose_index >= len(mols) or mols[pose_index] is None:
                logger.warning(f"Pose {pose_index} not found in {source_path}")
                return False
            
            mol = mols[pose_index]
            
            # Write to new SDF
            writer = Chem.SDWriter(str(dest_path))
            writer.write(mol)
            writer.close()
            
            return True
            
        elif suffix == '.pdbqt':
            # Convert PDBQT to SDF via PDB intermediate
            return self._pdbqt_to_sdf(source_path, dest_path, pose_index)
        
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return False
    
    def _pdbqt_to_sdf(self,
                      pdbqt_path: Path,
                      sdf_path: Path,
                      model_index: int = 0) -> bool:
        """Convert PDBQT to SDF.
        
        Args:
            pdbqt_path: Path to PDBQT file
            sdf_path: Destination SDF path
            model_index: Model index for multi-model PDBQT
            
        Returns:
            True if successful
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError("RDKit required: pip install rdkit")
        
        try:
            import openbabel.openbabel as ob
            from openbabel import pybel
            
            # Use OpenBabel for PDBQT conversion
            mols = list(pybel.readfile('pdbqt', str(pdbqt_path)))
            
            if model_index >= len(mols):
                logger.warning(f"Model {model_index} not found in {pdbqt_path}")
                return False
            
            mol = mols[model_index]
            mol.write('sdf', str(sdf_path), overwrite=True)
            
            return True
            
        except ImportError:
            # Fallback: parse PDBQT manually
            return self._parse_pdbqt_to_sdf(pdbqt_path, sdf_path, model_index)
    
    def _parse_pdbqt_to_sdf(self,
                            pdbqt_path: Path,
                            sdf_path: Path,
                            model_index: int = 0) -> bool:
        """Parse PDBQT manually and convert to SDF.
        
        This is a fallback when OpenBabel is not available.
        
        Args:
            pdbqt_path: Path to PDBQT file
            sdf_path: Destination SDF path
            model_index: Model index
            
        Returns:
            True if successful
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError("RDKit required")
        
        # Parse PDBQT to get coordinates and atom types
        models = []
        current_model = []
        
        with open(pdbqt_path, 'r') as f:
            for line in f:
                if line.startswith('MODEL'):
                    current_model = []
                elif line.startswith('ENDMDL'):
                    if current_model:
                        models.append(current_model)
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    current_model.append(line)
        
        # If no MODEL/ENDMDL, treat entire file as one model
        if not models and current_model:
            models = [current_model]
        
        if model_index >= len(models):
            logger.warning(f"Model {model_index} not found in {pdbqt_path}")
            return False
        
        # Create temporary PDB
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            for line in models[model_index]:
                # Convert PDBQT line to PDB format
                pdb_line = line[:66].rstrip() + '\n'
                tmp.write(pdb_line)
            tmp_path = tmp.name
        
        try:
            mol = Chem.MolFromPDBFile(tmp_path, removeHs=False)
            if mol is None:
                logger.warning(f"Failed to parse {pdbqt_path}")
                return False
            
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()
            
            return True
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def export_pose(self,
                    receptor_path: Path,
                    ligand_path: Path,
                    affinity: float,
                    pose_index: int = 0,
                    cluster_id: Optional[str] = None,
                    ligand_name: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None
                    ) -> Optional[ExportedSystem]:
        """Export a single pose to FLOWR directory structure.
        
        Args:
            receptor_path: Path to receptor PDB
            ligand_path: Path to ligand file (SDF or PDBQT)
            affinity: Binding affinity score
            pose_index: Pose index in multi-pose file
            cluster_id: Source cluster identifier
            ligand_name: Original ligand name
            metadata: Additional information
            
        Returns:
            ExportedSystem if successful, None otherwise
        """
        system_id = self._get_next_system_id()
        system_dir = self.config.output_dir / system_id
        
        # Check if exists
        if system_dir.exists():
            if not self.config.overwrite:
                logger.warning(f"System {system_id} already exists, skipping")
                # Return existing system info instead of None
                return ExportedSystem(
                    system_id=system_id,
                    system_dir=system_dir,
                    protein_path=system_dir / f'{system_id}_protein.pdb',
                    ligand_path=system_dir / f'{system_id}_ligand.sdf',
                    affinity=affinity,
                    cluster_id=cluster_id,
                    ligand_name=ligand_name,
                    pose_index=pose_index,
                    metadata=metadata or {}
                )
            shutil.rmtree(system_dir)
        
        system_dir.mkdir(parents=True)
        
        # Output paths
        protein_dest = system_dir / f'{system_id}_protein.pdb'
        ligand_dest = system_dir / f'{system_id}_ligand.sdf'
        
        # Copy/link receptor
        try:
            if self.config.copy_receptor:
                shutil.copy2(receptor_path, protein_dest)
            else:
                protein_dest.symlink_to(receptor_path.resolve())
        except Exception as e:
            logger.error(f"Failed to copy receptor: {e}")
            shutil.rmtree(system_dir, ignore_errors=True)
            return None
        
        # Extract/convert ligand pose
        try:
            if ligand_path.suffix.lower() == '.sdf' and pose_index == 0:
                # Simple case: just copy
                if self.config.copy_ligand:
                    shutil.copy2(ligand_path, ligand_dest)
                else:
                    ligand_dest.symlink_to(ligand_path.resolve())
            else:
                # Need to extract specific pose
                if not self._extract_pose_to_sdf(ligand_path, ligand_dest, pose_index):
                    shutil.rmtree(system_dir, ignore_errors=True)
                    return None
        except Exception as e:
            logger.error(f"Failed to process ligand: {e}")
            shutil.rmtree(system_dir, ignore_errors=True)
            return None
        
        return ExportedSystem(
            system_id=system_id,
            system_dir=system_dir,
            protein_path=protein_dest,
            ligand_path=ligand_dest,
            affinity=affinity,
            cluster_id=cluster_id,
            ligand_name=ligand_name,
            pose_index=pose_index,
            metadata=metadata or {}
        )
    
    def export_from_rescoring_results(self,
                                       results: List[Any],  # List[RescoringResult]
                                       cluster_pdb_map: Dict[str, Path]
                                       ) -> List[ExportedSystem]:
        """Export poses from rescoring results.
        
        Args:
            results: List of RescoringResult objects from rescoring module
            cluster_pdb_map: Mapping of cluster names to PDB paths
            
        Returns:
            List of successfully exported systems
        """
        exported = []
        
        for result in results:
            # Get receptor path from cluster map
            cluster_id = result.receptor_name
            if cluster_id not in cluster_pdb_map:
                logger.warning(f"Cluster {cluster_id} not found in PDB map")
                continue
            
            receptor_path = cluster_pdb_map[cluster_id]
            
            # Get affinity score
            affinity = result.scores.get('CNNaffinity', 
                        result.scores.get('affinity', 
                        result.primary_score or 0.0))
            
            # Export
            system = self.export_pose(
                receptor_path=receptor_path,
                ligand_path=result.pose.pose_file,
                affinity=affinity,
                pose_index=result.pose.pose_index,
                cluster_id=cluster_id,
                ligand_name=result.pose.ligand_name,
                metadata={
                    'scores': result.scores,
                    'original_metadata': result.metadata
                }
            )
            
            if system is not None:
                exported.append(system)
        
        logger.info(f"Exported {len(exported)} systems from {len(results)} results")
        return exported
    
    def _get_sdf_score(self, sdf_path: Path, score_key: str) -> float:
        """Read score from SDF tags."""
        try:
            from rdkit import Chem
            # Use SDMolSupplier to read the first molecule's tags
            suppl = Chem.SDMolSupplier(str(sdf_path))
            if len(suppl) > 0 and suppl[0] is not None:
                mol = suppl[0]
                if mol.HasProp(score_key):
                    return float(mol.GetProp(score_key))
        except Exception:
            pass
        return 0.0

    def export_from_directory(self,
                              docking_dir: Path,
                              cluster_dir: Path,
                              score_file: Optional[Path] = None,
                              score_threshold: float = 0.0,
                              affinity_key: str = 'CNNaffinity'
                              ) -> List[ExportedSystem]:
        """Export poses from docking output directory.
        
        Args:
            docking_dir: Directory with docking outputs per cluster
            cluster_dir: Directory with cluster PDB files
            score_file: Optional CSV/JSON with scores
            score_threshold: Minimum score to export
            affinity_key: Key for affinity in score file
            
        Returns:
            List of exported systems
        """
        exported = []
        
        # Build cluster PDB map
        cluster_pdb_map = {}
        for pdb in cluster_dir.glob('cluster_*.pdb'):
            cluster_name = pdb.stem
            cluster_pdb_map[cluster_name] = pdb
        
        # Load scores if provided
        scores = {}
        if score_file and score_file.exists():
            scores = self._load_scores(score_file, affinity_key)
        
        # Process each cluster subdirectory
        for cluster_subdir in sorted(docking_dir.iterdir()):
            if not cluster_subdir.is_dir():
                continue
            
            cluster_name = cluster_subdir.name
            if cluster_name not in cluster_pdb_map:
                continue
            
            receptor_path = cluster_pdb_map[cluster_name]
            
            # Find pose files
            for pose_file in cluster_subdir.glob('*.sdf'):
                if not pose_file.name.endswith('.sdf'):
                    continue
                    
                ligand_name = pose_file.stem.replace('_out', '').replace('_gnina', '')
                
                # Get score
                score_key = f"{cluster_name}/{ligand_name}"
                if scores:
                    affinity = scores.get(score_key, 0.0)
                else:
                    # Try to read from SDF
                    affinity = self._get_sdf_score(pose_file, affinity_key)
                
                # Check threshold
                # Note: This assumes "higher is better" if threshold is positive (pKd)
                # and "lower is better" if threshold is negative (binding energy)
                if score_threshold > 0:
                    # pKd or CNNscore: higher is better
                    if affinity < score_threshold:
                        continue
                else:
                    # Binding energy: lower is better
                    if affinity > score_threshold:
                        continue
                
                system = self.export_pose(
                    receptor_path=receptor_path,
                    ligand_path=pose_file,
                    affinity=affinity,
                    pose_index=0,
                    cluster_id=cluster_name,
                    ligand_name=ligand_name
                )
                
                if system is not None:
                    exported.append(system)
        
        return exported
    
    def _load_scores(self, 
                     score_file: Path,
                     affinity_key: str) -> Dict[str, float]:
        """Load scores from file."""
        scores = {}
        
        if score_file.suffix == '.json':
            with open(score_file) as f:
                data = json.load(f)
                for entry in data:
                    key = f"{entry.get('cluster')}/{entry.get('ligand')}"
                    scores[key] = entry.get(affinity_key, 0.0)
        
        elif score_file.suffix == '.csv':
            import csv
            with open(score_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = f"{row.get('cluster')}/{row.get('ligand')}"
                    scores[key] = float(row.get(affinity_key, 0.0))
        
        return scores
    
    def save_manifest(self,
                      exported: List[ExportedSystem],
                      manifest_path: Optional[Path] = None):
        """Save manifest of exported systems.
        
        Args:
            exported: List of ExportedSystem objects
            manifest_path: Output path (default: output_dir/manifest.json)
        """
        if manifest_path is None:
            manifest_path = self.config.output_dir / 'manifest.json'
        
        manifest = {
            'n_systems': len(exported),
            'systems': [s.to_dict() for s in exported]
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Saved manifest to {manifest_path}")
    
    @classmethod
    def load_manifest(cls, manifest_path: Path) -> List[ExportedSystem]:
        """Load manifest of exported systems.
        
        Args:
            manifest_path: Path to manifest JSON
            
        Returns:
            List of ExportedSystem objects
        """
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        systems = []
        for s in manifest['systems']:
            systems.append(ExportedSystem(
                system_id=s['system_id'],
                system_dir=Path(s['system_dir']),
                protein_path=Path(s['protein_path']),
                ligand_path=Path(s['ligand_path']),
                affinity=s['affinity'],
                cluster_id=s.get('cluster_id'),
                ligand_name=s.get('ligand_name'),
                pose_index=s.get('pose_index', 0),
                metadata=s.get('metadata', {})
            ))
        
        return systems
