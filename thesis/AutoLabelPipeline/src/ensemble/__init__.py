"""
Ensemble module for post-hoc aggregation of model predictions.
Enables majority voting to combine predictions without re-running inference.
"""

from .loader import EnsembleLoader, FramePredictionDataset
from .aggregator import (
    MajorityVoter,
    EnsembleAggregator,
)
from .evaluator import EnsembleEvaluator
from .disagreement import DisagreementAnalyzer

__all__ = [
    "EnsembleLoader",
    "FramePredictionDataset",
    "MajorityVoter",
    "EnsembleAggregator",
    "EnsembleEvaluator",
    "DisagreementAnalyzer",
]
