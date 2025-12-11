
import numpy as np
import lmdb
import os
import argparse

def create_splits(lmdb_path, output_path):
    env = lmdb.open(lmdb_path, subdir=True, readonly=True, lock=False)
    with env.begin() as txn:
        length_bytes = txn.get(b'__len__')
        if length_bytes:
            length = int(length_bytes.decode())
        else:
            # Fallback if __len__ is missing (shouldn't happen with our writer)
            length = int(txn.stat()['entries'])
    
    print(f"Found {length} data entries in {lmdb_path}")
    
    indices = np.arange(length)
    # For small dataset, just use all for everything or split simply
    # Since we have 1 sample, we must put it in all or we can't run train/val/test
    
    train_idx = indices
    val_idx = indices
    test_idx = indices
    
    np.savez(output_path, idx_train=train_idx, idx_val=val_idx, idx_test=test_idx)
    print(f"Saved splits to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    create_splits(args.lmdb_path, args.output_path)
