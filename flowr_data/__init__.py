"""
FLOWR.ROOT Data Engineering Module.

This module provides utilities for converting filtered docking results
into the LMDB format required by FLOWR.ROOT for training.

The pipeline:
    1. Export filtered poses to standardized directory structure
    2. Featurize protein and ligand structures
    3. Serialize features into LMDB database
    4. Compute data statistics for prior distribution

Typical usage:
    >>> from flowr_data import FlowrDataPreparer, FeaturizationConfig
    >>> config = FeaturizationConfig(...)
    >>> preparer = FlowrDataPreparer(config)
    >>> preparer.prepare_from_rescoring_results(results)
    >>> preparer.generate_lmdb()
"""

from .preparer import FlowrDataPreparer, FlowrDataConfig
from .featurizer import (
    ProteinFeaturizer,
    LigandFeaturizer,
    FeaturizationConfig,
    SystemFeatures
)
from .lmdb_writer import LMDBWriter, DataStatistics
from .exporter import PoseExporter, ExportConfig

__all__ = [
    'FlowrDataPreparer',
    'FlowrDataConfig',
    'ProteinFeaturizer',
    'LigandFeaturizer',
    'FeaturizationConfig',
    'SystemFeatures',
    'LMDBWriter',
    'DataStatistics',
    'PoseExporter',
    'ExportConfig',
]
