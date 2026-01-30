"""
Output generation module for turn signal detection pipeline.
Handles all output formats, visualizations, and reporting.
"""
from .formatters import (
    CSVFormatter,
    JSONFormatter,
    COCOFormatter,
    ReviewQueueFormatter,
    save_predictions
)
from .visualizer import (
    FrameVisualizer,
    VideoVisualizer,
    visualize_samples
)
from .metrics_reporter import (
    MetricsReporter,
    generate_and_save_report
)
from .output_generator import (
    OutputGenerator,
    create_output_generator,
    save_labels_only,
    generate_quick_report
)

__all__ = [
    # Formatters
    'CSVFormatter',
    'JSONFormatter',
    'COCOFormatter',
    'ReviewQueueFormatter',
    'save_predictions',
    
    # Visualizers
    'FrameVisualizer',
    'VideoVisualizer',
    'visualize_samples',
    
    # Metrics
    'MetricsReporter',
    'generate_and_save_report',
    
    # Main generator
    'OutputGenerator',
    'create_output_generator',
    'save_labels_only',
    'generate_quick_report',
]
