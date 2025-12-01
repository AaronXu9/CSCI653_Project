"""
Docking module for high-throughput molecular docking.

This module provides an extensible framework for docking ligand libraries
against protein conformational ensembles (e.g., BioEmu clusters).

Supported Engines:
    - UniDock: GPU-accelerated AutoDock Vina variant
    - (Future) AutoDock-GPU, Glide, etc.
"""

from .base import DockingEngine, DockingConfig, DockingResult
from .unidock import UniDockEngine, UniDockConfig
from .utils import prepare_receptor_pdbqt, prepare_ligand_pdbqt, detect_binding_site

__all__ = [
    'DockingEngine',
    'DockingConfig', 
    'DockingResult',
    'UniDockEngine',
    'UniDockConfig',
    'prepare_receptor_pdbqt',
    'prepare_ligand_pdbqt',
    'detect_binding_site',
]
