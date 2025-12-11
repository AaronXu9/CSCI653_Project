
import lmdb
import pickle
import argparse
import sys
import os
from tqdm import tqdm

# Add flowr_root to path
sys.path.append(os.path.join(os.getcwd(), 'flowr_root'))

from flowr.util.pocket import PocketComplex

def add_lengths(lmdb_path):
    env = lmdb.open(lmdb_path, subdir=True, readonly=False, lock=False, map_size=1099511627776) # 1TB
    
    lengths_full = []
    lengths_no_pocket_hs = []
    lengths_no_ligand_pocket_hs = []
    
    with env.begin(write=True) as txn:
        # Get length
        length_bytes = txn.get(b'__len__')
        if length_bytes:
            length = int(length_bytes.decode())
        else:
            length = int(txn.stat()['entries'])
            # Note: entries includes metadata keys, so this might be wrong if metadata exists
            # But we are adding metadata now.
            # If __len__ is missing, we should count keys that are integers.
            # But our writer put __len__.
        
        print(f"Processing {length} entries...")
        
        for i in tqdm(range(length)):
            key = str(i).encode('utf-8')
            data = txn.get(key)
            if data is None:
                print(f"Warning: Key {key} not found")
                continue
                
            complex_system = PocketComplex.from_bytes(data)
            
            # Calculate lengths
            l_full = complex_system.seq_length
            
            # remove_hs returns a new PocketComplex
            # include_ligand=False means remove Hs from protein only?
            # No, remove_hs(include_ligand=False) means remove Hs from protein, keep ligand Hs?
            # Let's check PocketComplex.remove_hs signature if possible, but based on preprocess.py:
            # length_no_pocket_hs = complex_system.remove_hs(include_ligand=False).seq_length
            # length_no_ligand_pocket_hs = complex_system.remove_hs(include_ligand=True).seq_length
            
            l_no_pocket_hs = complex_system.remove_hs(include_ligand=False).seq_length
            l_no_ligand_pocket_hs = complex_system.remove_hs(include_ligand=True).seq_length
            
            lengths_full.append(l_full)
            lengths_no_pocket_hs.append(l_no_pocket_hs)
            lengths_no_ligand_pocket_hs.append(l_no_ligand_pocket_hs)
            
        # Save metadata
        txn.put(b'lengths_full', pickle.dumps(lengths_full))
        txn.put(b'lengths_no_pocket_hs', pickle.dumps(lengths_no_pocket_hs))
        txn.put(b'lengths_no_ligand_pocket_hs', pickle.dumps(lengths_no_ligand_pocket_hs))
        
    env.close()
    print("Lengths added to LMDB.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, required=True)
    args = parser.parse_args()
    
    add_lengths(args.lmdb_path)
