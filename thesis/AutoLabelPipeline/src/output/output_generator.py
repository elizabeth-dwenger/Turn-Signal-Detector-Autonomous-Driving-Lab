"""
Output generation orchestrator.
Coordinates saving predictions, creating visualizations, and generating reports.
"""
from pathlib import Path
from typing import Dict, List
import json
import logging
from datetime import datetime

from .formatters import save_predictions, SequenceFormatter, ReviewQueueFormatter
from .visualizer import create_visualizer
from utils.enums import OutputFormat


logger = logging.getLogger(__name__)


class OutputGenerator:
    """
    Orchestrates all output generation.
    Saves predictions, creates visualizations, and generates reports.
    """
    
    def __init__(self, output_config, experiment_config):
        self.config = output_config
        self.output_dir = Path(experiment_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer if needed
        if output_config.save_visualizations:
            self.visualizer = create_visualizer(output_config)
        else:
            self.visualizer = None
    
    def save_sequence_predictions(self, sequence_id: str,
                                  predictions: List[Dict],
                                  quality_report: Dict = None) -> Dict:
        """
        Save predictions for a single sequence.
        """
        # Clean sequence_id for filenames
        safe_id = sequence_id.replace('/', '_').replace('\\', '_')
        
        output_files = {}
        
        # Save in each requested format
        for format in self.config.formats:
            format_str = format.value
            filename = f"{safe_id}.{format_str if format_str != 'coco' else 'json'}"
            file_path = self.output_dir / filename
            
            # Call save_predictions with minimal kwargs
            # Each formatter handles what it needs
            if format_str == 'csv':
                from .formatters import CSVFormatter
                CSVFormatter.save(predictions, str(file_path))
            elif format_str == 'json':
                from .formatters import JSONFormatter
                JSONFormatter.save(
                    predictions,
                    str(file_path),
                    metadata={'sequence_id': sequence_id},
                    include_raw_output=self.config.include_raw_output
                )
            elif format_str == 'coco':
                from .formatters import COCOFormatter
                COCOFormatter.save(
                    predictions,
                    str(file_path),
                    sequence_info={'sequence_id': sequence_id}
                )
            
            output_files[format_str] = str(file_path)
        
        # Save review queue if frames are flagged
        if self.config.export_review_queue and quality_report:
            if quality_report.get('flagged_frames'):
                review_path = self.output_dir / f"{safe_id}_review_queue.json"
                ReviewQueueFormatter.save(
                    quality_report['flagged_frames'],
                    str(review_path),
                    sequence_id=sequence_id
                )
                output_files['review_queue'] = str(review_path)
        
        return output_files
    
    def save_dataset_predictions(self, sequences_predictions: Dict[str, Dict]) -> Dict:
        """
        Save predictions for entire dataset.
        """
        logger.info(f"Saving predictions for {len(sequences_predictions)} sequences")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_sequences': len(sequences_predictions),
            'output_directory': str(self.output_dir),
            'formats': [f.value for f in self.config.formats],
            'sequences': {}
        }
        
        # Save each sequence
        for sequence_id, data in sequences_predictions.items():
            output_files = self.save_sequence_predictions(
                sequence_id,
                data['predictions'],
                data.get('quality_report')
            )
            
            summary['sequences'][sequence_id] = {
                'output_files': output_files,
                'num_predictions': len(data['predictions']),
                'flagged_frames': data.get('quality_report', {}).get('total_flagged', 0)
            }
        
        # Save summary
        summary_path = self.output_dir / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved dataset summary: {summary_path}")
        
        return summary
    
    def create_visualizations(self, sequences_data: Dict, sample_rate: float = None):
        """
        Create visualizations for sequences.
        """
        if not self.visualizer:
            logger.info("Visualizations disabled in config")
            return
        
        effective_sample_rate = self.config.visualization_sample_rate if sample_rate is None else sample_rate
        logger.info(f"Creating visualizations (sample rate: {effective_sample_rate})")
        
        self.visualizer.create_sample_visualizations(
            sequences_data,
            sample_rate=effective_sample_rate
        )
    
    def generate_report(self, dataset_stats: Dict, model_metrics: Dict = None) -> str:
        """
        Generate comprehensive report.
        """
        report_path = self.output_dir / 'pipeline_report.json'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_statistics': dataset_stats,
            'output_directory': str(self.output_dir),
            'configuration': {
                'formats': [f.value for f in self.config.formats],
                'visualizations_enabled': self.config.save_visualizations
            }
        }
        
        if model_metrics:
            report['model_metrics'] = model_metrics
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated pipeline report: {report_path}")
        
        # Also create human-readable text report
        text_report_path = self.output_dir / 'pipeline_report.txt'
        self._generate_text_report(report, text_report_path)
        
        return str(report_path)
    
    def _generate_text_report(self, report: Dict, output_path: str):
        """Generate human-readable text report"""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TURN SIGNAL DETECTION PIPELINE REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Output Directory: {report['output_directory']}\n\n")
            
            # Dataset statistics
            f.write("DATASET STATISTICS\n")
            f.write("-"*80 + "\n")
            stats = report['dataset_statistics']
            f.write(f"Total Sequences: {stats.get('total_sequences', 'N/A')}\n")
            f.write(f"Total Frames: {stats.get('total_frames', 'N/A')}\n")
            f.write(f"Frames Flagged: {stats.get('total_flagged', 'N/A')}\n\n")
            
            if 'label_distribution' in stats:
                f.write("Label Distribution:\n")
                for label, count in stats['label_distribution'].items():
                    pct = count / stats['total_frames'] * 100 if stats['total_frames'] > 0 else 0
                    f.write(f"  {label:.<20} {count:>6} ({pct:>5.1f}%)\n")
            
            # Model metrics
            if 'model_metrics' in report:
                f.write("\n\nMODEL METRICS\n")
                f.write("-"*80 + "\n")
                metrics = report['model_metrics']
                for key, value in metrics.items():
                    f.write(f"{key:.<40} {value}\n")
            
            # Configuration
            f.write("\n\nCONFIGURATION\n")
            f.write("-"*80 + "\n")
            config = report['configuration']
            f.write(f"Output Formats: {', '.join(config['formats'])}\n")
            f.write(f"Visualizations: {config['visualizations_enabled']}\n")
        
        logger.info(f"Generated text report: {output_path}")


def create_output_generator(config):
    """
    Factory function to create output generator.
    """
    return OutputGenerator(config.output, config.experiment)
