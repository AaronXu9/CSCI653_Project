
import json
import pickle
import argparse
from pathlib import Path
from rdkit import Chem

def create_rdkit_mols(manifest_path, output_path):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    mols = []
    for system in manifest['systems']:
        ligand_path = system['ligand_path']
        # Handle relative paths if necessary, but manifest seems to have relative paths from workspace root?
        # Or relative to where prepare_flowr_data was run.
        # I'll assume paths are correct relative to CWD or absolute.
        
        if not os.path.exists(ligand_path):
            # Try relative to manifest dir
            ligand_path = Path(manifest_path).parent.parent / 'raw' / Path(ligand_path).name
            # Wait, manifest path is .../raw/manifest.json
            # system['ligand_path'] is test_output/flowr_data/raw/system_0001/system_0001_ligand.sdf
            # If I run from workspace root, it should be fine.
        
        mol = Chem.MolFromMolFile(str(ligand_path), sanitize=False) # Sanitize=False to avoid issues, but we might want it sanitized
        if mol is None:
            print(f"Warning: Failed to load molecule from {ligand_path}")
        else:
            try:
                Chem.SanitizeMol(mol)
            except:
                print(f"Warning: Failed to sanitize molecule from {ligand_path}")
        
        # We need to store the mol as bytes or just the mol object?
        # scriptutil.py: rdkit_mols = pickle.load(f) -> list of mols (or bytes?)
        # train_mols = [Chem.Mol(rdkit_mols[i]) for i in train_ids]
        # So rdkit_mols[i] is passed to Chem.Mol().
        # Chem.Mol() copy constructor takes a Mol.
        # So rdkit_mols should be a list of Mol objects.
        # But pickling Mol objects can be tricky across versions.
        # Usually one pickles the PropertyMol or just Mol.
        # Let's pickle the Mol object.
        mols.append(mol)
        
    with open(output_path, 'wb') as f:
        pickle.dump(mols, f)
    
    print(f"Saved {len(mols)} RDKit mols to {output_path}")

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    create_rdkit_mols(args.manifest_path, args.output_path)
