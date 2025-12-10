import mdtraj as md
import numpy as np
from sklearn.cluster import KMeans
import os
import glob

def main():
    # Configuration
    output_dir = 'bioemu_clusters'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load topology
    # We use the provided topology file. 
    # Note: The topology has 2134 atoms, but the npz has 430 positions.
    # We assume the npz positions correspond to Alpha Carbons (CA).
    full_topology = md.load('test_output/protein_ensemble/topology.pdb').topology
    ca_topology = full_topology.subset(full_topology.select("name CA"))
    
    print(f"Full topology atoms: {full_topology.n_atoms}")
    print(f"CA topology atoms: {ca_topology.n_atoms}")
    
    # Load BioEmu output files
    npz_files = sorted(glob.glob('test_output/protein_ensemble/*.npz'))
    # Limit to 100 structures for testing as requested
    npz_files = npz_files[:100]
    
    print(f"Processing {len(npz_files)} files...")
    
    xyz_list = []
    valid_files = []
    
    for f in npz_files:
        try:
            data = np.load(f)
            # pos shape is (1, 430, 3)
            pos = data['pos']
            if pos.shape[1] != ca_topology.n_atoms:
                print(f"Warning: {f} has shape {pos.shape}, expected (1, {ca_topology.n_atoms}, 3). Skipping.")
                continue
            xyz_list.append(pos)
            valid_files.append(f)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not xyz_list:
        print("No valid structures found.")
        return

    # Concatenate all frames
    xyz = np.vstack(xyz_list)
    print(f"Total frames: {xyz.shape[0]}")
    
    # Create trajectory
    traj = md.Trajectory(xyz, ca_topology)
    
    # Define pocket
    # Using the logic provided: resid 99 to 110 or resid 145 to 155
    # Note: 'resid' in mdtraj selects by 0-based index.
    pocket_inds = traj.topology.select("resid 99 to 110 or resid 145 to 155")
    print(f"Selected {len(pocket_inds)} atoms for clustering.")
    
    if len(pocket_inds) == 0:
        print("Error: No atoms selected for pocket. Check residue indices.")
        return

    # Slice trajectory to pocket for clustering
    t_pocket = traj.atom_slice(pocket_inds)
    
    # Compute Pairwise RMSD Matrix
    # We align to the first frame of the pocket trajectory before computing RMSD?
    # Or just compute pairwise RMSD. mdtraj.rmsd computes RMSD to a reference.
    # We need a matrix (N, N).
    
    n_frames = traj.n_frames
    distances = np.empty((n_frames, n_frames))
    
    print("Computing RMSD matrix...")
    for i in range(n_frames):
        # Compute RMSD of all frames relative to frame i
        # atom_slice already creates a new trajectory, so we can use it directly
        # We must align the structures on the pocket atoms for the RMSD to be meaningful for clustering shapes
        # t_pocket.superpose(t_pocket, frame=i) -> this modifies t_pocket in place? 
        # No, superpose returns self. But we want pairwise RMSD.
        # md.rmsd(target, reference) aligns automatically? No, it assumes pre-aligned or just computes RMSD.
        # Usually one superposes first.
        # But if we want pairwise RMSD matrix, we need to be careful.
        # Let's use the standard approach: Superpose everything to frame 0 first to remove global rotation/translation?
        # But we are clustering based on internal conformation of the pocket.
        # So we should align the pocket.
        
        # Actually, let's just compute RMSD between frame i and all other frames.
        # md.rmsd aligns the target to the reference if precentered=False (default).
        # Wait, md.rmsd does NOT perform rotation/translation alignment by default?
        # Documentation says: "Compute the RMSD... of each frame in target to a reference structure."
        # It does NOT say it optimizes rotation.
        # To get optimal RMSD (least squares), we need to superpose.
        
        # Efficient way:
        # 1. Superpose all frames to frame 0 (based on pocket atoms).
        # 2. Then compute RMSD? No, pairwise RMSD requires pairwise alignment.
        # However, aligning all to a common reference is a good approximation if the structures are not too diverse.
        # But for rigorous pairwise RMSD, we need to align i and j for every pair.
        # That's O(N^2) alignments. For N=100 it's fine (10000 ops).
        
        # Let's do it properly for N=100.
        # But mdtraj doesn't have a pairwise_rmsd function that does alignment.
        # We can use t_pocket.slice(i) as reference.
        
        # Optimization: Align all to frame 0 first.
        t_pocket.superpose(t_pocket, frame=0)
        # Now they are in a common frame of reference.
        # The RMSD between i and j in this common frame is an upper bound on the optimal RMSD.
        # For clustering, this is often sufficient and much faster.
        # The user text says "Compute RMSD Matrix".
        
        distances[i] = md.rmsd(t_pocket, t_pocket, frame=i)

    # Clustering
    n_clusters = min(50, n_frames) # Handle case if we have < 50 frames
    print(f"Clustering into {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(distances)
    
    # Extract Centroids
    # We want to find the frame closest to each cluster center.
    # The cluster centers are in "distance space" (if we clustered on the distance matrix).
    # kmeans.cluster_centers_ has shape (n_clusters, n_features) = (50, 100).
    # We find the sample i such that distances[i] is closest to center[k].
    
    labels = kmeans.labels_
    cluster_indices = []
    
    for k in range(n_clusters):
        # Get indices of points in this cluster
        members = np.where(labels == k)[0]
        if len(members) == 0:
            continue
            
        # Get the center
        center = kmeans.cluster_centers_[k]
        
        # Find member closest to center
        # We compare the distance vector of the member (distances[member]) to the center vector.
        dists_to_center = np.linalg.norm(distances[members] - center, axis=1)
        closest_member_idx = members[np.argmin(dists_to_center)]
        cluster_indices.append(closest_member_idx)
        
        # Save the PDB
        # We save the FULL structure (from 'traj'), not just the pocket
        save_path = os.path.join(output_dir, f"cluster_{k}.pdb")
        traj[closest_member_idx].save(save_path)
        
        # Post-process to remove MODEL/ENDMDL tags which confuse Uni-Dock
        with open(save_path, 'r') as f:
            lines = f.readlines()
        with open(save_path, 'w') as f:
            for line in lines:
                if not line.startswith('MODEL') and not line.startswith('ENDMDL'):
                    f.write(line)
        
    print(f"Saved {len(cluster_indices)} cluster representatives to {output_dir}")

if __name__ == "__main__":
    main()
