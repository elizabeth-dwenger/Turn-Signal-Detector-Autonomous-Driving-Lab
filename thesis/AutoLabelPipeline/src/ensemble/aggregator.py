"""
Ensemble aggregation with voting mechanisms.
Implements majority vote for combining predictions from multiple models.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VotingResult:
    """Result of a voting operation"""
    label: str
    confidence: float
    agreement_count: int  # Number of models voting for this label
    total_models: int
    voting_distribution: Dict[str, int] = None  # label -> count


class MajorityVoter:
    """Simple majority vote (>50% agreement)"""
    
    def __init__(self, tie_break_method: str = "confidence"):
        self.tie_break_method = tie_break_method
    
    def vote(self, predictions: List, **kwargs) -> VotingResult:
        """
        Majority vote: sum votes per label, return label with >50%.
        """
        if not predictions:
            return VotingResult(
                label="none",
                confidence=0.0,
                agreement_count=0,
                total_models=0,
                voting_distribution={},
            )
        
        # Count votes per label
        vote_counts = {}
        confidence_per_label = {}
        
        for pred in predictions:
            vote_counts[pred.label] = vote_counts.get(pred.label, 0) + 1
            
            if pred.label not in confidence_per_label:
                confidence_per_label[pred.label] = []
            confidence_per_label[pred.label].append(pred.confidence)
        
        total = len(predictions)
        majority_threshold = total / 2.0
        
        # Check for majority
        majority_labels = [
            label for label, count in vote_counts.items()
            if count > majority_threshold
        ]
        
        if len(majority_labels) == 1:
            label = majority_labels[0]
            agreement = vote_counts[label]
            avg_confidence = np.mean(confidence_per_label[label])
            
            return VotingResult(
                label=label,
                confidence=avg_confidence,
                agreement_count=agreement,
                total_models=total,
                voting_distribution=vote_counts,
            )
        
        best_label = max(
            confidence_per_label.keys(),
            key=lambda l: np.mean(confidence_per_label[l])
        )
        
        avg_confidence = np.mean(confidence_per_label[best_label])
        agreement = vote_counts[best_label]
        
        return VotingResult(
            label=best_label,
            confidence=avg_confidence,
            agreement_count=agreement,
            total_models=total,
            voting_distribution=vote_counts,
        )


class EnsembleAggregator:
    """Orchestrates ensemble prediction aggregation using majority voting"""
    def __init__(self, voter: MajorityVoter = None):
        self.voter = voter if voter is not None else MajorityVoter()
    
    def aggregate(self, frame_predictions_dict: Dict,
                  verbose: bool = False) -> pd.DataFrame:
        """
        Apply majority voting to all frames.
        """
        results = []
        
        for (seq_id, frame_id), predictions in frame_predictions_dict.items():
            voting_result = self.voter.vote(predictions)
            
            results.append({
                "sequence_id": seq_id,
                "frame_id": frame_id,
                "ensemble_label": voting_result.label,
                "ensemble_confidence": voting_result.confidence,
                "agreement_count": voting_result.agreement_count,
                "total_models": voting_result.total_models,
                "voting_distribution": voting_result.voting_distribution,
            })
            
            if verbose and len(results) <= 5:
                logger.info(
                    f"Frame {seq_id}:{frame_id} -> {voting_result.label} "
                    f"(conf={voting_result.confidence:.2f}, "
                    f"agree={voting_result.agreement_count}/{voting_result.total_models})"
                )
        
        df = pd.DataFrame(results)
        logger.info(f"Aggregated {len(df)} frames using majority vote")
        
        return df
