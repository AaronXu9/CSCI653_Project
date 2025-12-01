
# Dynamic Generative Structure-Based Drug Design Pipeline

## Overview
[Diagram for the Dynamic Generative Structure-Based Drug Design Pipeline](figs/diagram.svg)

<img src="figs/diagram.svg" alt="Dynamic Generative Structure-Based Drug Design Pipeline">

[cite/]This repository houses a state-of-the-art computational pipeline for Structure-Based Drug Design (SBDD) that transitions from traditional static methods toward a dynamic, probabilistic understanding of molecular recognition[cite: 3].

[cite_start]Traditional SBDD relies on static crystal structures, which often represent single, low-energy minima and miss transient, bioactive conformations such as "cryptic" pockets [cite: 5-7]. [cite_start]To overcome this "static trap," this pipeline integrates biomolecular emulation with geometric deep learning [cite: 15-17].

[cite_start]We utilize **BioEmu** to generate thermodynamic ensembles of the target protein directly from sequence, capturing functional motions and rare states[cite: 26, 52]. [cite_start]These ensembles are used to create a synthetic training dataset to fine-tune **FLOWR.ROOT**, an SE(3)-equivariant flow matching generative model[cite: 14, 177]. [cite_start]This results in a generator explicitly tailored to the dynamic conformational landscape of the target protein[cite: 15].

## Pipeline Workflow

The following diagram outlines the four-phase process implemented in this repository:


---

## Detailed Functionality

### Phase I: Constructing the Target Ensemble with BioEmu
[cite_start]Instead of relying on a single PDB file, we generate a high-quality structural ensemble that reflects the protein's solution-state dynamics[cite: 46].
* [cite_start]**BioEmu Sampling:** We use BioEmu trained on aggregate MD data to predict the equilibrium distribution of structures from the amino acid sequence[cite: 26, 51]. [cite_start]We typically generate 5,000+ samples to populate the tails of the distribution where cryptic states reside[cite: 64].
* [cite_start]**Refinement:** Raw samples undergo sidechain repacking and NVT equilibration (using OpenMM) to resolve subtle clashes and ensure physical viability[cite: 75, 82].
* [cite_start]**Ensemble Reduction:** To create a diverse, representative training set, refined structures are clustered based on binding pocket RMSD using MDTraj, reducing the ensemble to approximately 50 distinct representative states[cite: 88, 96, 99].

### Phase II: Generating Synthetic Training Data
FLOWR.ROOT requires protein-ligand pairs for supervised training. [cite_start]We generate a high-fidelity "Synthetic Holo-Set" by docking diverse libraries into our ensemble representatives[cite: 110, 118].
* [cite_start]**Massive Batch Docking:** We utilize **Uni-Dock** for its extreme GPU-accelerated throughput to execute large-scale virtual screening against all cluster representatives[cite: 126, 137].
* [cite_start]**High-Fidelity Rescoring:** Generated poses are rescored using **Gnina's** deep learning CNN scoring function, which is superior at distinguishing real biological binding modes from artifacts[cite: 127, 128].
* [cite_start]**Data Engineering:** Poses are filtered for high CNN scores (>0.9) and predicted affinity, then featurized and serialized into LMDB format for high-performance I/O during training[cite: 146, 148, 149].

### Phase III: Fine-Tuning FLOWR.ROOT
[cite_start]We adapt the generalist FLOWR.ROOT foundation model to the specific geometric and electrostatic boundary conditions of the target protein's dynamic pocket[cite: 42, 44].
* [cite_start]**Flow Matching Architecture:** FLOWR.ROOT uses Continuous Normalizing Flows to learn a time-dependent vector field that transports a prior noise distribution to valid ligand structures, respecting SE(3)-equivariance[cite: 34, 37].
* [cite_start]**Low-Rank Adaptation (LoRA):** To prevent catastrophic forgetting of general chemical rules, we apply LoRA adapters to the model's cross-attention layers while freezing the backbone weights[cite: 193, 195].
* [cite_start]**Joint Training:** The model is trained with a combined loss function, optimizing both the flow matching objective for structure generation and MSE loss for the joint affinity prediction head[cite: 200, 204].

### Phase IV: Inference, Steering, and Validation
[cite_start]Once fine-tuned, the model serves as a bespoke generator for the target, followed by rigorous physics-based validation[cite: 220].
* [cite_start]**Affinity Steering:** During inference, we generate thousands of trajectories by solving the ODE and use the trained affinity head to perform importance sampling, prioritizing high-affinity "super-binders"[cite: 222, 226, 231].
* [cite_start]**PoseBusters Validation:** Generated ligands are subjected to the **PoseBusters** suite to ensure chemical validity, planarity, correct bond geometry, and the absence of severe protein-ligand clashes[cite: 234, 236].
* **Redocking Consistency:** As a final orthogonal check, valid ligands are redocked using Gnina. [cite_start]A low Self-Consistency RMSD (< 2.0 Å) indicates the generative model found a stable energy minimum supported by physics-based docking[cite: 248, 253].

## Software Stack

[cite_start]This pipeline relies on the following key software components[cite: 270]:

| Component | Software | Role |
| :--- | :--- | :--- |
| Dynamics | **BioEmu** | Protein Ensemble Generation |
| Clustering | **MDTraj** | Ensemble Reduction & Selection |
| Docking | **Uni-Dock** (GPU) | High-Throughput Pose Generation |
| Rescoring | **Gnina** | High-Fidelity Affinity Filtering |
| Generator | **FLOWR.ROOT** | Ligand Design & Affinity Prediction |
| Validation| **PoseBusters** | Physical & Chemical Sanity Checks |