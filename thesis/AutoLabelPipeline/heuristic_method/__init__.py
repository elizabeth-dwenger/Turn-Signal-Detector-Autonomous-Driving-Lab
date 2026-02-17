"""
Heuristic-based turn signal detection module.

This module provides a computational (non-VLM) approach to turn signal detection using yellow channel isolation and FFT-based periodic signal analysis.

Usage:
    # As a module
    from heuristic_method import HeuristicDetector, test_heuristic
    
    detector = HeuristicDetector(fps=5.0)
    result = detector.predict_video(video_tensor)
    
    # Or run the full pipeline
    test_heuristic(
        config_path="config.yaml",
        test_sequences_file="test_sequences.json"
    )
    
    # CLI usage
    python -m heuristic_method.test_heuristic \\
        --config config.yaml \\
        --test-sequences test_sequences.json \\
        --output-dir prompt_comparison
"""

from .heuristic_detector import HeuristicDetector, load_images_from_paths
from .test_heuristic import test_heuristic

__all__ = [
    "HeuristicDetector",
    "load_images_from_paths",
    "test_heuristic",
]
