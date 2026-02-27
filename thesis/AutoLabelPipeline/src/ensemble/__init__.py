"""
Ensemble module for post-hoc aggregation of model predictions.
Enables majority voting to combine predictions without re-running inference.
Also provides agreement-based filtering for two-model consensus.
"""

from .loader import EnsembleLoader, FramePredictionDataset
from .aggregator import (
    MajorityVoter,
    EnsembleAggregator,
)
from .evaluator import EnsembleEvaluator
from .disagreement import DisagreementAnalyzer
from .agreement_filter import (
    AgreementFilter,
    FilteringReport,
    HeuristicResultLoader,
    VLMResultLoader,
)

__all__ = [
    "EnsembleLoader",
    "FramePredictionDataset",
    "MajorityVoter",
    "EnsembleAggregator",
    "EnsembleEvaluator",
    "DisagreementAnalyzer",
    "AgreementFilter",
    "FilteringReport",
    "HeuristicResultLoader",
    "VLMResultLoader",
]
