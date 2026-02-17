#!/usr/bin/env python
"""
Test script for ensemble module.
Demonstrates basic workflow on synthetic data.
"""
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ensemble.loader import EnsembleLoader, FramePrediction, FramePredictionDataset
from ensemble.aggregator import (
    MajorityVoter,
    EnsembleAggregator,
)
from ensemble.evaluator import EnsembleEvaluator
from ensemble.disagreement import DisagreementAnalyzer


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_synthetic_dataset():
    """
    Create synthetic dataset for testing.
    Simulates 3 models predicting on 100 frames from 2 sequences.
    """
    dataset = FramePredictionDataset()
    
    # Ground truth: some frames are left, some right, some none
    np.random.seed(42)
    
    sequences = ["seq_001", "seq_002"]
    num_frames = 50
    
    for seq_id in sequences:
        for frame_id in range(num_frames):
            # Generate ground truth
            gt_label = np.random.choice(["left", "right", "none"], p=[0.2, 0.2, 0.6])
            dataset.ground_truth[(seq_id, frame_id)] = gt_label
            
            # Simulate 3 models making predictions
            models = ["model_a", "model_b", "model_c"]
            predictions = []
            
            for model_idx, model_name in enumerate(models):
                # Model A is good overall
                if model_name == "model_a":
                    if np.random.random() < 0.8:
                        label = gt_label
                        confidence = np.random.uniform(0.7, 0.95)
                    else:
                        # Occasionally make mistakes
                        label = np.random.choice(["left", "right", "hazard", "none"])
                        confidence = np.random.uniform(0.4, 0.6)
                
                # Model B is mediocre, but good on rare cases
                elif model_name == "model_b":
                    if gt_label != "none" and np.random.random() < 0.6:
                        label = gt_label
                        confidence = np.random.uniform(0.6, 0.85)
                    elif gt_label == "none" and np.random.random() < 0.7:
                        label = gt_label
                        confidence = np.random.uniform(0.6, 0.85)
                    else:
                        label = np.random.choice(["left", "right", "hazard", "none"])
                        confidence = np.random.uniform(0.3, 0.5)
                
                # Model C is worse but diverse
                else:
                    if np.random.random() < 0.5:
                        label = gt_label
                        confidence = np.random.uniform(0.5, 0.8)
                    else:
                        label = np.random.choice(["left", "right", "hazard", "none"])
                        confidence = np.random.uniform(0.2, 0.5)
                
                predictions.append(
                    FramePrediction(
                        frame_id=frame_id,
                        label=label,
                        confidence=confidence,
                        model_name=model_name,
                    )
                )
            
            # Store in dataset
            key = (seq_id, frame_id)
            dataset.predictions[key] = predictions
    
    return dataset


def test_loader():
    """Test EnsembleLoader functionality"""
    print("\n" + "="*60)
    print("TEST 1: EnsembleLoader")
    print("="*60)
    
    # Create synthetic dataset
    dataset = create_synthetic_dataset()
    
    # Validate
    loader = EnsembleLoader()
    report = loader.validate_dataset(dataset)
    
    print(f"\nDataset Statistics:")
    print(f"  Total frames: {report['num_frames']}")
    print(f"  Models: {report['models']}")
    print(f"  Label distribution: {report['label_distribution']}")
    print(f"  Confidence stats: {report['confidence_stats']}")
    
    # Convert to DataFrame
    df = loader.to_dataframe(dataset)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    
    return dataset


def test_majority_vote(dataset):
    """Test MajorityVoter"""
    print("\n" + "="*60)
    print("TEST 2: Majority Vote")
    print("="*60)
    
    voter = MajorityVoter(tie_break_method="confidence")
    aggregator = EnsembleAggregator(voter)
    
    ensemble_df = aggregator.aggregate(dataset.predictions, verbose=True)
    
    print(f"\nEnsemble predictions (first 10):")
    print(ensemble_df.head(10))
    
    print(f"\nEnsemble predictions (first 10):")
    print(ensemble_df.head(10))
    
    return ensemble_df


def test_evaluator(dataset, ensemble_df):
    """Test EnsembleEvaluator"""
    print("\n" + "="*60)
    print("TEST 5: EnsembleEvaluator")
    print("="*60)
    
    evaluator = EnsembleEvaluator()
    
    # Prepare ground truth and predictions
    y_true = []
    y_pred = []
    
    for _, row in ensemble_df.iterrows():
        key = (row["sequence_id"], row["frame_id"])
        if key in dataset.ground_truth:
            y_true.append(dataset.ground_truth[key])
            y_pred.append(row["ensemble_label"])
    
    # Compute metrics
    metrics = evaluator.compute_frame_metrics(y_true, y_pred)
    
    print(f"\nFrame-Level Metrics:")
    print(f"  Macro F1: {metrics['frame_macro_f1']:.3f}")
    print(f"  F1 (left): {metrics['f1_left']:.3f}")
    print(f"  F1 (right): {metrics['f1_right']:.3f}")
    print(f"  F1 (hazard): {metrics['f1_hazard']:.3f}")
    print(f"  F1 (none): {metrics['f1_none']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")


def test_disagreement_analyzer(dataset):
    """Test DisagreementAnalyzer"""
    print("\n" + "="*60)
    print("TEST 6: DisagreementAnalyzer")
    print("="*60)
    
    analyzer = DisagreementAnalyzer()
    
    # Compute disagreement metrics
    metrics = analyzer.compute_disagreement_metrics(dataset.predictions)
    
    print(f"\nDisagreement Statistics:")
    print(f"  Total frames: {metrics['total_frames']}")
    print(f"  Unanimous frames: {metrics['unanimous_frames']}")
    print(f"  High-disagreement frames: {metrics['high_disagreement_frames']}")
    print(f"  Mean entropy: {metrics.get('mean_entropy', 'N/A'):.3f}")
    
    # Identify disagreements
    disagreements = analyzer.identify_disagreement_frames(
        dataset.predictions, entropy_threshold=0.5
    )
    
    print(f"\nHigh-disagreement frames (entropy > 0.5): {len(disagreements)}")
    if disagreements:
        print(f"  Example: {disagreements[0]}")
    
    # Categorize patterns
    patterns = analyzer.categorize_patterns(disagreements)
    
    print(f"\nDisagreement patterns:")
    for pattern_str, summary in sorted(
        patterns.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:5]:
        print(f"  {summary['pattern']}: {summary['count']} frames")
    
    # Error correlation
    error_analysis = analyzer.analyze_error_correlation(
        dataset.predictions, dataset.ground_truth
    )
    
    print(f"\nModel Error Profiles:")
    for model_name, profile in error_analysis['model_profiles'].items():
        print(f"  {model_name}:")
        print(f"    Accuracy: {profile['accuracy']:.3f}")
        print(f"    False positives: {profile['false_positives']}")
        print(f"    False negatives: {profile['false_negatives']}")


def main():
    """Run all tests"""
    setup_logging()
    
    print("\n" + "="*60)
    print("ENSEMBLE MODULE TEST SUITE")
    print("="*60)
    
    # Test 1: Loader
    dataset = test_loader()
    
    # Test 2: Majority Vote
    ensemble_mv = test_majority_vote(dataset)
    
    # Test 3: Confidence-Weighted Vote
    ensemble_cwv = test_confidence_weighted_vote(dataset)
    
    # Test 4: Consensus Threshold Vote
    ensemble_ctv = test_consensus_threshold_vote(dataset)
    
    # Test 5: Evaluator
    test_evaluator(dataset, ensemble_mv)
    
    # Test 6: Disagreement Analyzer
    test_disagreement_analyzer(dataset)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
