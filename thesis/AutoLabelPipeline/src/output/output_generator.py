"""
Main output generation orchestrator.
Coordinates all output formats, visualizations, and reports.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from .formatters import (
    save_predictions, CSVFormatter, JSONFormatter,
    COCOFormatter, ReviewQueueFormatter
)
from .visualizer import visualize_samples, FrameVisualizer, VideoVisualizer
from .metrics_reporter import generate_and_save_report, MetricsReporter


logger = logging.getLogger(__name__)


class OutputGenerator:
    """
    Orchestrates all output generation tasks.
    Handles file formats, visualizations, metrics, and review queues.
    """
    
    def __init__(self, output_config, experiment_config):
        """
        Args:
            output_config: OutputConfig from configuration
            experiment_config: ExperimentConfig for metadata
        """
        self.config = output_config
        self.experiment_config = experiment_config
        self.output_dir = Path(experiment_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track what was generated
        self.generated_files = {
            'labels': {},
            'visualizations': [],
            'reports': {},
            'review_queue': None
        }
    
    def generate_all_outputs(self,
                            predictions_by_sequence: Dict[str, List[Dict]],
                            quality_reports: Optional[Dict[str, Dict]] = None,
                            images_by_sequence: Optional[Dict[str, List]] = None,
                            model_metrics: Optional[Dict] = None,
                            dataset_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate all configured outputs.
        
        Args:
            predictions_by_sequence: Dict mapping sequence_id to predictions list
            quality_reports: Optional quality control reports per sequence
            images_by_sequence: Optional images for visualization
            model_metrics: Optional model performance metrics
            dataset_metadata: Optional dataset information
        
        Returns:
            Dict summarizing all generated outputs
        """
        logger.info("Starting output generation...")
        
        # 1. Generate label files
        logger.info("Generating label files...")
        self._generate_label_files(predictions_by_sequence, dataset_metadata)
        
        # 2. Generate visualizations
        if self.config.save_visualizations and images_by_sequence:
            logger.info("Generating visualizations...")
            self._generate_visualizations(predictions_by_sequence, images_by_sequence)
        else:
            logger.info("Skipping visualizations (disabled or no images)")
        
        # 3. Generate metrics report
        logger.info("Generating metrics report...")
        self._generate_metrics_report(
            predictions_by_sequence,
            quality_reports,
            model_metrics
        )
        
        # 4. Generate review queue
        if self.config.export_review_queue and quality_reports:
            logger.info("Generating review queue...")
            self._generate_review_queue(predictions_by_sequence, quality_reports)
        else:
            logger.info("Skipping review queue (disabled or no quality reports)")
        
        logger.info("Output generation complete!")
        
        return self.get_summary()
    
    def _generate_label_files(self,
                             predictions_by_sequence: Dict[str, List[Dict]],
                             metadata: Optional[Dict] = None):
        """Generate label files in configured formats"""
        
        # Build metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'experiment': self.experiment_config.name,
            'output_dir': str(self.output_dir),
            'timestamp': datetime.now().isoformat()
        })
        
        formats = [f.value for f in self.config.formats]
        
        config_dict = {
            'include_confidence': self.config.include_confidence,
            'include_raw_output': self.config.include_raw_output,
            'metadata': metadata
        }
        
        # Save in all requested formats
        output_files = save_predictions(
            predictions_by_sequence,
            str(self.output_dir),
            formats=formats,
            config=config_dict
        )
        
        self.generated_files['labels'] = output_files
        
        logger.info(f"Generated label files: {list(output_files.keys())}")
    
    def _generate_visualizations(self,
                                predictions_by_sequence: Dict[str, List[Dict]],
                                images_by_sequence: Dict[str, List]):
        """Generate visualization outputs"""
        
        viz_dir = self.config.visualization_output_dir or (self.output_dir / 'visualizations')
        viz_dir = Path(viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sample visualizations
        output_files = visualize_samples(
            predictions_by_sequence,
            images_by_sequence,
            str(viz_dir),
            sample_rate=self.config.visualization_sample_rate,
            format='images'  # Can be made configurable
        )
        
        self.generated_files['visualizations'] = output_files
        
        logger.info(f"Generated {len(output_files)} visualization files in {viz_dir}")
    
    def _generate_metrics_report(self,
                                predictions_by_sequence: Dict[str, List[Dict]],
                                quality_reports: Optional[Dict[str, Dict]],
                                model_metrics: Optional[Dict]):
        """Generate metrics report"""
        
        report_path = self.output_dir / 'metrics_report.json'
        
        # Determine model name
        model_name = getattr(self.experiment_config, 'description', 'Unknown Model')
        dataset_name = str(self.output_dir.name)
        
        saved_path = generate_and_save_report(
            predictions_by_sequence,
            str(report_path),
            model_name=model_name,
            dataset_name=dataset_name,
            quality_reports=quality_reports,
            model_metrics=model_metrics,
            print_summary=True
        )
        
        self.generated_files['reports']['metrics'] = saved_path
        
        logger.info(f"Generated metrics report: {saved_path}")
    
    def _generate_review_queue(self,
                              predictions_by_sequence: Dict[str, List[Dict]],
                              quality_reports: Dict[str, Dict]):
        """Generate review queue for manual annotation"""
        
        review_path = self.output_dir / f'review_queue.{self.config.review_queue_format}'
        
        ReviewQueueFormatter.save(
            quality_reports,
            predictions_by_sequence,
            str(review_path)
        )
        
        self.generated_files['review_queue'] = str(review_path)
        
        logger.info(f"Generated review queue: {review_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all generated outputs.
        
        Returns:
            Dict with file paths and statistics
        """
        return {
            'output_directory': str(self.output_dir),
            'generated_files': self.generated_files,
            'file_count': {
                'label_files': len(self.generated_files['labels']),
                'visualizations': len(self.generated_files['visualizations']),
                'reports': len(self.generated_files['reports']),
                'review_queue': 1 if self.generated_files['review_queue'] else 0
            }
        }
    
    def print_summary(self):
        """Print human-readable summary of outputs"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("OUTPUT GENERATION SUMMARY")
        print("="*70)
        print(f"\nOutput Directory: {summary['output_directory']}")
        
        print("\nLabel Files:")
        for format_name, path in self.generated_files['labels'].items():
            print(f"  {format_name:8s}: {path}")
        
        if self.generated_files['visualizations']:
            print(f"\nVisualizations: {len(self.generated_files['visualizations'])} files")
            print(f"  Location: {Path(self.generated_files['visualizations'][0]).parent}")
        
        print("\nReports:")
        for report_name, path in self.generated_files['reports'].items():
            print(f"  {report_name}: {path}")
        
        if self.generated_files['review_queue']:
            print(f"\nReview Queue: {self.generated_files['review_queue']}")
        
        print("\n" + "="*70)


def create_output_generator(config):
    """
    Factory function to create OutputGenerator from PipelineConfig.
    
    Args:
        config: PipelineConfig instance
    
    Returns:
        OutputGenerator instance
    """
    return OutputGenerator(config.output, config.experiment)


# Convenience functions for standalone use

def save_labels_only(predictions_by_sequence: Dict[str, List[Dict]],
                     output_dir: str,
                     formats: List[str] = None) -> Dict[str, str]:
    """
    Quick function to just save label files.
    
    Args:
        predictions_by_sequence: Predictions dict
        output_dir: Output directory
        formats: List of formats ('csv', 'json', 'coco')
    
    Returns:
        Dict mapping format to file path
    """
    if formats is None:
        formats = ['csv', 'json']
    
    return save_predictions(predictions_by_sequence, output_dir, formats)


def generate_quick_report(predictions_by_sequence: Dict[str, List[Dict]],
                         output_path: str,
                         model_name: str = "Model") -> str:
    """
    Quick function to generate just a metrics report.
    
    Args:
        predictions_by_sequence: Predictions dict
        output_path: Output JSON path
        model_name: Model identifier
    
    Returns:
        Path to saved report
    """
    return generate_and_save_report(
        predictions_by_sequence,
        output_path,
        model_name=model_name,
        dataset_name="Dataset"
    )

