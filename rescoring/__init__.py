"""
Rescoring module for pose refinement and filtering.

This module provides an extensible framework for rescoring docking poses
using various methods (CNN-based, physics-based, ML-based).

Supported Rescoring Methods:
    - Gnina: CNN-based rescoring for pose quality assessment
    - (Future) RF-Score, OnionNet, DeepDTA, etc.
"""

from .base import RescoringEngine, RescoringConfig, RescoringResult, PoseData
from .gnina import GninaRescorer, GninaConfig
from .filters import ScoreFilter, FilterConfig, FilterResult

__all__ = [
    'RescoringEngine',
    'RescoringConfig',
    'RescoringResult',
    'PoseData',
    'GninaRescorer',
    'GninaConfig',
    'ScoreFilter',
    'FilterConfig',
    'FilterResult',
]
