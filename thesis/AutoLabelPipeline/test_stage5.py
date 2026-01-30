"""
Unit tests for Stage 5 output generation.
Tests formatters, visualizers, metrics, and the main generator.

python test_stage5.py
"""
import sys
from pathlib import Path
import json
import csv
import numpy as np
import tempfile
import shutil
from typing import List, Dict, Tuple


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.output.formatters import (
    CSVFormatter, JSONFormatter, COCOFormatter,
    ReviewQueueFormatter, save_predictions
)
from src.output.metrics_reporter import MetricsReporter, generate_and_save_report
from src.output.output_generator import OutputGenerator
from src.utils.config import OutputConfig, ExperimentConfig, OutputFormat


# ============================================================================
# Helper Functions
# ============================================================================

def create_mock_predictions(n_frames: int = 10) -> List[Dict]:
    """Create mock predictions for testing"""
    predictions = []
    labels = ['none', 'left', 'right'] * (n_frames // 3 + 1)
    
    for i in range(n_frames):
        predictions.append({
            'frame_id': i,
            'label': labels[i],
            'confidence': 0.8 + (i % 3) * 0.05,
            'raw_output': f'Mock output {i}',
            'smoothed': (i % 5 == 0),
            'original_label': labels[i] if (i % 5 != 0) else labels[(i+1) % len(labels)]
        })
    
    return predictions


def create_mock_images(n_frames: int = 10, size: Tuple[int, int] = (480, 640)) -> List[np.ndarray]:
    """Create mock images for testing"""
    images = []
    for i in range(n_frames):
        # Create simple gradient image
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img[:, :] = [i * 10 % 255, 100, 150]
        images.append(img)
    return images


def create_mock_quality_report(n_frames: int = 10) -> Dict:
    """Create mock quality report"""
    return {
        'total_frames': n_frames,
        'total_flagged': 2,
        'flagged_frames': [
            {'frame_index': 2, 'flags': ['low_confidence']},
            {'frame_index': 5, 'flags': ['rapid_changes']}
        ],
        'confidence_stats': {
            'mean': 0.85,
            'median': 0.87,
            'std': 0.1
        }
    }


def create_output_config(**kwargs) -> OutputConfig:
    """Create OutputConfig with defaults"""
    defaults = {
        'formats': [OutputFormat.CSV, OutputFormat.JSON],
        'include_confidence': True,
        'include_raw_output': False,
        'save_visualizations': True,
        'visualization_sample_rate': 0.1,
        'visualization_output_dir': None,
        'export_review_queue': True,
        'review_queue_format': 'json'
    }
    defaults.update(kwargs)
    return OutputConfig(**defaults)


def create_experiment_config(output_dir: str) -> ExperimentConfig:
    """Create ExperimentConfig"""
    return ExperimentConfig(
        name='test_experiment',
        output_dir=output_dir,
        random_seed=42,
        description='Test experiment for output generation'
    )


# ============================================================================
# Formatter Tests
# ============================================================================

class TestCSVFormatter:
    """Test CSV output formatting"""
    
    def test_basic_csv_generation(self):
        """Test basic CSV file creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {
                'seq1': create_mock_predictions(5)
            }
            
            output_path = Path(tmpdir) / 'test.csv'
            CSVFormatter.save(predictions, str(output_path))
            
            assert output_path.exists()
            
            # Read and verify
            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 5
            assert 'sequence_id' in rows[0]
            assert 'label' in rows[0]
            assert 'confidence' in rows[0]
    
    def test_csv_without_metadata(self):
        """Test CSV generation without metadata columns"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {
                'seq1': create_mock_predictions(3)
            }
            
            output_path = Path(tmpdir) / 'test.csv'
            CSVFormatter.save(predictions, str(output_path), include_metadata=False)
            
            # Read and verify
            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Should not have metadata columns
            assert 'smoothed' not in rows[0]
            assert 'original_label' not in rows[0]


class TestJSONFormatter:
    """Test JSON output formatting"""
    
    def test_basic_json_generation(self):
        """Test basic JSON file creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {
                'seq1': create_mock_predictions(5),
                'seq2': create_mock_predictions(3)
            }
            
            output_path = Path(tmpdir) / 'test.json'
            JSONFormatter.save(predictions, str(output_path))
            
            assert output_path.exists()
            
            # Read and verify
            with open(output_path) as f:
                data = json.load(f)
            
            assert 'sequences' in data
            assert 'seq1' in data['sequences']
            assert 'seq2' in data['sequences']
            assert data['sequences']['seq1']['num_frames'] == 5
            assert data['sequences']['seq2']['num_frames'] == 3
    
    def test_json_with_metadata(self):
        """Test JSON with custom metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {'seq1': create_mock_predictions(3)}
            
            metadata = {
                'model': 'test_model',
                'version': '1.0'
            }
            
            output_path = Path(tmpdir) / 'test.json'
            JSONFormatter.save(predictions, str(output_path), metadata=metadata)
            
            with open(output_path) as f:
                data = json.load(f)
            
            assert data['metadata']['model'] == 'test_model'
            assert data['metadata']['version'] == '1.0'


class TestCOCOFormatter:
    """Test COCO format output"""
    
    def test_coco_format_structure(self):
        """Test COCO format structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {'seq1': create_mock_predictions(5)}
            
            output_path = Path(tmpdir) / 'test_coco.json'
            COCOFormatter.save(predictions, str(output_path))
            
            assert output_path.exists()
            
            with open(output_path) as f:
                data = json.load(f)
            
            # Check COCO structure
            assert 'info' in data
            assert 'images' in data
            assert 'annotations' in data
            assert 'categories' in data
            
            # Check categories
            assert len(data['categories']) == 4
            category_names = {c['name'] for c in data['categories']}
            assert category_names == {'none', 'left', 'right', 'both'}
            
            # Check annotations match images
            assert len(data['images']) == len(data['annotations'])


class TestReviewQueueFormatter:
    """Test review queue generation"""
    
    def test_review_queue_generation(self):
        """Test review queue file creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {'seq1': create_mock_predictions(10)}
            quality_reports = {'seq1': create_mock_quality_report(10)}
            
            output_path = Path(tmpdir) / 'review.json'
            ReviewQueueFormatter.save(quality_reports, predictions, str(output_path))
            
            assert output_path.exists()
            
            with open(output_path) as f:
                data = json.load(f)
            
            assert 'total_flagged' in data
            assert 'frames' in data
            assert data['total_flagged'] == 2  # From mock report
            assert len(data['frames']) == 2


# ============================================================================
# Metrics Reporter Tests
# ============================================================================

class TestMetricsReporter:
    """Test metrics report generation"""
    
    def test_basic_report_generation(self):
        """Test basic metrics report"""
        predictions = {
            'seq1': create_mock_predictions(20),
            'seq2': create_mock_predictions(15)
        }
        
        reporter = MetricsReporter('TestModel', 'TestDataset')
        report = reporter.generate_report(predictions)
        
        # Check structure
        assert 'metadata' in report
        assert 'dataset_stats' in report
        assert 'label_distribution' in report
        assert 'confidence_stats' in report
        assert 'quality_metrics' in report
        assert 'temporal_metrics' in report
        
        # Check dataset stats
        assert report['dataset_stats']['total_sequences'] == 2
        assert report['dataset_stats']['total_frames'] == 35
    
    def test_label_distribution(self):
        """Test label distribution calculation"""
        # Create predictions with known distribution
        predictions = {
            'seq1': [
                {'label': 'none', 'confidence': 0.9},
                {'label': 'none', 'confidence': 0.9},
                {'label': 'left', 'confidence': 0.9},
                {'label': 'right', 'confidence': 0.9},
            ]
        }
        
        reporter = MetricsReporter('TestModel', 'TestDataset')
        report = reporter.generate_report(predictions)
        
        dist = report['label_distribution']
        assert dist['none']['count'] == 2
        assert dist['left']['count'] == 1
        assert dist['right']['count'] == 1
        assert dist['none']['percentage'] == 50.0
    
    def test_confidence_statistics(self):
        """Test confidence statistics calculation"""
        predictions = {
            'seq1': [
                {'label': 'left', 'confidence': 0.5},
                {'label': 'left', 'confidence': 0.7},
                {'label': 'left', 'confidence': 0.9},
            ]
        }
        
        reporter = MetricsReporter('TestModel', 'TestDataset')
        report = reporter.generate_report(predictions)
        
        cs = report['confidence_stats']
        assert cs['min'] == 0.5
        assert cs['max'] == 0.9
        assert abs(cs['mean'] - 0.7) < 0.01  # Approximately 0.7
    
    def test_temporal_metrics(self):
        """Test temporal metrics calculation"""
        # Create sequence with clear episodes
        predictions = {
            'seq1': [
                {'label': 'none', 'confidence': 0.9},
                {'label': 'left', 'confidence': 0.9},
                {'label': 'left', 'confidence': 0.9},
                {'label': 'left', 'confidence': 0.9},
                {'label': 'none', 'confidence': 0.9},
                {'label': 'right', 'confidence': 0.9},
                {'label': 'right', 'confidence': 0.9},
                {'label': 'none', 'confidence': 0.9},
            ]
        }
        
        reporter = MetricsReporter('TestModel', 'TestDataset')
        report = reporter.generate_report(predictions)
        
        tm = report['temporal_metrics']
        assert tm['signal_episodes_detected'] == 2  # One left, one right
        assert tm['avg_signal_duration_frames'] == 2.5  # (3 + 2) / 2
    
    def test_report_save(self):
        """Test saving report to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {'seq1': create_mock_predictions(10)}
            
            reporter = MetricsReporter('TestModel', 'TestDataset')
            report = reporter.generate_report(predictions)
            
            output_path = Path(tmpdir) / 'report.json'
            saved_path = reporter.save_report(report, str(output_path))
            
            assert Path(saved_path).exists()
            
            # Verify can be read back
            with open(saved_path) as f:
                loaded_report = json.load(f)
            
            assert loaded_report['metadata']['model'] == 'TestModel'


# ============================================================================
# Output Generator Tests
# ============================================================================

class TestOutputGenerator:
    """Test integrated output generation"""
    
    def test_label_file_generation(self):
        """Test generating all label file formats"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {
                'seq1': create_mock_predictions(10),
                'seq2': create_mock_predictions(5)
            }
            
            output_config = create_output_config(
                formats=[OutputFormat.CSV, OutputFormat.JSON, OutputFormat.COCO]
            )
            experiment_config = create_experiment_config(tmpdir)
            
            generator = OutputGenerator(output_config, experiment_config)
            generator._generate_label_files(predictions)
            
            # Check files were created
            assert (Path(tmpdir) / 'predictions.csv').exists()
            assert (Path(tmpdir) / 'predictions.json').exists()
            assert (Path(tmpdir) / 'predictions_coco.json').exists()
    
    def test_metrics_report_generation(self):
        """Test metrics report generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {'seq1': create_mock_predictions(10)}
            quality_reports = {'seq1': create_mock_quality_report(10)}
            
            output_config = create_output_config()
            experiment_config = create_experiment_config(tmpdir)
            
            generator = OutputGenerator(output_config, experiment_config)
            generator._generate_metrics_report(predictions, quality_reports, None)
            
            assert (Path(tmpdir) / 'metrics_report.json').exists()
    
    def test_review_queue_generation(self):
        """Test review queue generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {'seq1': create_mock_predictions(10)}
            quality_reports = {'seq1': create_mock_quality_report(10)}
            
            output_config = create_output_config()
            experiment_config = create_experiment_config(tmpdir)
            
            generator = OutputGenerator(output_config, experiment_config)
            generator._generate_review_queue(predictions, quality_reports)
            
            assert (Path(tmpdir) / 'review_queue.json').exists()
    
    def test_complete_output_generation(self):
        """Test complete output generation pipeline"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {
                'seq1': create_mock_predictions(10),
                'seq2': create_mock_predictions(5)
            }
            quality_reports = {
                'seq1': create_mock_quality_report(10),
                'seq2': create_mock_quality_report(5)
            }
            
            # Don't test visualizations (no images)
            output_config = create_output_config(save_visualizations=False)
            experiment_config = create_experiment_config(tmpdir)
            
            generator = OutputGenerator(output_config, experiment_config)
            summary = generator.generate_all_outputs(
                predictions,
                quality_reports=quality_reports
            )
            
            # Check summary
            assert 'output_directory' in summary
            assert 'generated_files' in summary
            assert summary['file_count']['label_files'] > 0
            assert summary['file_count']['reports'] > 0
    
    def test_output_summary(self):
        """Test output summary generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions = {'seq1': create_mock_predictions(10)}
            
            output_config = create_output_config(save_visualizations=False)
            experiment_config = create_experiment_config(tmpdir)
            
            generator = OutputGenerator(output_config, experiment_config)
            generator.generate_all_outputs(predictions)
            
            summary = generator.get_summary()
            
            assert 'output_directory' in summary
            assert 'generated_files' in summary
            assert 'file_count' in summary


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test end-to-end output generation"""
    
    def test_full_pipeline(self):
        """Test complete output generation pipeline"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create realistic data
            predictions = {}
            quality_reports = {}
            
            for i in range(3):
                seq_id = f'sequence_{i}'
                predictions[seq_id] = create_mock_predictions(20)
                quality_reports[seq_id] = create_mock_quality_report(20)
            
            # Configure output
            output_config = create_output_config(
                formats=[OutputFormat.CSV, OutputFormat.JSON],
                save_visualizations=False,  # No images available
                export_review_queue=True
            )
            experiment_config = create_experiment_config(tmpdir)
            
            # Generate outputs
            generator = OutputGenerator(output_config, experiment_config)
            summary = generator.generate_all_outputs(
                predictions,
                quality_reports=quality_reports
            )
            
            # Verify all outputs
            output_dir = Path(tmpdir)
            assert (output_dir / 'predictions.csv').exists()
            assert (output_dir / 'predictions.json').exists()
            assert (output_dir / 'metrics_report.json').exists()
            assert (output_dir / 'review_queue.json').exists()
            
            # Verify summary
            assert summary['file_count']['label_files'] == 2
            assert summary['file_count']['reports'] == 1


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == '__main__':
    try:
        import pytest
        sys.exit(pytest.main([__file__, '-v', '--tb=short']))
    except ImportError:
        print("pytest not found, running basic tests...")
        
        test_classes = [
            TestCSVFormatter(),
            TestJSONFormatter(),
            TestCOCOFormatter(),
            TestReviewQueueFormatter(),
            TestMetricsReporter(),
            TestOutputGenerator(),
            TestIntegration()
        ]
        
        passed = 0
        failed = 0
        
        for test_class in test_classes:
            class_name = test_class.__class__.__name__
            print(f"\n{'='*60}")
            print(f"Running {class_name}")
            print('='*60)
            
            test_methods = [m for m in dir(test_class) if m.startswith('test_')]
            
            for method_name in test_methods:
                try:
                    method = getattr(test_class, method_name)
                    method()
                    print(f" {method_name}")
                    passed += 1
                except Exception as e:
                    print(f" {method_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
        
        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed")
        print('='*60)
        
        sys.exit(0 if failed == 0 else 1)

