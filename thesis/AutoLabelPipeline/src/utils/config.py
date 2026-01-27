"""
Configuration management for turn signal detection pipeline.
Loads, validates, and provides access to configuration parameters.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from .enums import (
    ModelType, InferenceMode, TurnSignalLabel,
    SmoothingMethod, OutputFormat, LogLevel
)


@dataclass
class ExperimentConfig:
    """Experiment metadata and identification"""
    name: str
    output_dir: str
    random_seed: int = 42
    description: str = ""
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data input configuration"""
    # CSV file with tracking data (your format)
    input_csv: str
    
    # Directory containing crop images
    crop_base_dir: str
    
    # Directory containing full frame images (optional)
    frame_base_dir: Optional[str] = None
    
    # Filters
    max_sequences: Optional[int] = None  # Limit number of sequences to process
    sequence_filter: Optional[List[str]] = None  # Process only these sequence IDs
    
    # For video mode: how to construct video sequences
    video_fps: int = 10  # Assumed FPS for video reconstruction
    
    def __post_init__(self):
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"Input CSV not found: {self.input_csv}")
        if not os.path.exists(self.crop_base_dir):
            raise FileNotFoundError(f"Crop directory not found: {self.crop_base_dir}")


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration"""
    resize_resolution: List[int] = field(default_factory=lambda: [640, 480])  # [width, height]
    normalize: bool = True
    maintain_aspect_ratio: bool = True
    padding_color: List[int] = field(default_factory=lambda: [0, 0, 0])  # RGB for padding
    
    # For video models: sequence construction
    max_sequence_length: Optional[int] = None  # Max frames in a sequence (None = no limit)
    sequence_stride: int = 1  # Sample every Nth frame for long sequences
    
    def __post_init__(self):
        assert len(self.resize_resolution) == 2, "Resolution must be [width, height]"
        assert self.resize_resolution[0] > 0 and self.resize_resolution[1] > 0
        assert len(self.padding_color) == 3, "Padding color must be RGB [r, g, b]"


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    type: ModelType
    inference_mode: InferenceMode
    model_name_or_path: str
    device: str = "cuda"
    
    # Generation parameters
    batch_size: int = 1  # For video mode, typically 1 sequence at a time
    max_new_tokens: int = 100
    temperature: float = 0.0  # Deterministic
    top_p: float = 1.0
    do_sample: bool = False
    
    # Prompt
    prompt_template_path: str = "data/prompts/turn_signal_video.txt"
    
    # Model-specific settings (stored as dict for flexibility)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # For API-based models (Cosmos via API)
    api_key_env: Optional[str] = None  # Environment variable name for API key
    api_endpoint: Optional[str] = None
    api_rate_limit: Optional[int] = None  # Requests per minute
    
    def __post_init__(self):
        # Validate prompt template exists
        if not os.path.exists(self.prompt_template_path):
            raise FileNotFoundError(f"Prompt template not found: {self.prompt_template_path}")
        
        # Load API key if specified
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise ValueError(f"API key environment variable '{self.api_key_env}' not set")
            self.model_kwargs['api_key'] = api_key


@dataclass
class SingleImageConfig:
    """Configuration specific to single-image inference mode"""
    # Strategy for determining signal episodes from individual frame predictions
    min_signal_duration_frames: int = 3  # Min consecutive frames for valid signal
    max_gap_frames: int = 2  # Max frames of 'none' within a signal episode
    
    # Episode detection
    confidence_threshold_start: float = 0.7  # Confidence to START an episode
    confidence_threshold_continue: float = 0.5  # Confidence to CONTINUE an episode
    
    # Interpolation
    interpolate_gaps: bool = True  # Fill gaps within episodes


@dataclass
class PostprocessingConfig:
    """Post-processing configuration"""
    # Temporal smoothing
    temporal_smoothing_enabled: bool = True
    smoothing_method: SmoothingMethod = SmoothingMethod.MEDIAN
    smoothing_window_size: int = 7
    
    # Confidence filtering
    confidence_threshold: float = 0.6
    
    # Physical constraints
    min_signal_duration_frames: int = 3
    max_signal_duration_frames: Optional[int] = None  # None = no limit
    allow_both_signals: bool = False  # Flag 'both' as anomaly if False
    
    # For single-image mode
    single_image: Optional[SingleImageConfig] = None
    
    def __post_init__(self):
        if self.smoothing_window_size % 2 == 0:
            raise ValueError("Smoothing window size must be odd")


@dataclass
class QualityControlConfig:
    """Quality control and review flagging"""
    flag_low_confidence: bool = True
    low_confidence_threshold: float = 0.4
    
    # Random sampling for manual review
    random_sample_rate: float = 0.05  # 5%
    stratified_sampling: bool = True  # Equal samples per class
    
    # Anomaly detection
    flag_both_signals: bool = True
    flag_rapid_changes: bool = True
    rapid_change_threshold: int = 3  # Changes within N frames


@dataclass
class OutputConfig:
    """Output generation configuration"""
    formats: List[OutputFormat] = field(default_factory=lambda: [OutputFormat.CSV, OutputFormat.JSON])
    
    include_confidence: bool = True
    include_raw_output: bool = False  # Save model's raw text response
    
    # Visualization
    save_visualizations: bool = True
    visualization_sample_rate: float = 0.01  # Visualize 1%
    visualization_output_dir: Optional[str] = None  # Defaults to output_dir/visualizations
    
    # Review queue
    export_review_queue: bool = True
    review_queue_format: str = "json"  # For manual annotation tools


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    console_output: bool = True
    track_metrics: bool = True
    
    # Progress tracking
    show_progress_bar: bool = True
    log_frequency: int = 100  # Log every N frames/sequences


class PipelineConfig:
    """
    Main configuration class that aggregates all config sections.
    Loads from YAML file and provides validated access.
    """
    
    def __init__(self, config_path: str, override_params: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration from YAML file.
        
        config_path: Path to YAML configuration file
        override_params: Dict of parameters to override (for experiments)
        """
        self.config_path = config_path
        self._raw_config = self._load_yaml(config_path)
        
        # Apply overrides if provided
        if override_params:
            self._apply_overrides(override_params)
        
        # Parse into structured dataclasses
        self.experiment = self._parse_section(ExperimentConfig, 'experiment')
        self.data = self._parse_section(DataConfig, 'data')
        self.preprocessing = self._parse_section(PreprocessingConfig, 'preprocessing')
        self.model = self._parse_section(ModelConfig, 'model')
        self.postprocessing = self._parse_section(PostprocessingConfig, 'postprocessing')
        self.quality_control = self._parse_section(QualityControlConfig, 'quality_control')
        self.output = self._parse_section(OutputConfig, 'output')
        self.logging = self._parse_section(LoggingConfig, 'logging')
        
        # Additional validation
        self._validate_config()
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML file with environment variable substitution"""
        with open(path, 'r') as f:
            content = f.read()
        
        # Replace environment variables (${VAR_NAME} syntax)
        import re
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        content = re.sub(r'\$\{([^}]+)\}', replace_env_var, content)
        return yaml.safe_load(content)
    
    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply parameter overrides to raw config"""
        def update_nested(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = update_nested(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self._raw_config = update_nested(self._raw_config, overrides)
    
    def _parse_section(self, dataclass_type, section_name: str):
        """Parse a config section into its dataclass"""
        section_data = self._raw_config.get(section_name, {})
        
        # Handle nested dataclasses (e.g., single_image config)
        if section_name == 'postprocessing' and 'single_image' in section_data:
            single_image_data = section_data.pop('single_image')
            # Only create SingleImageConfig if data is not None
            if single_image_data is not None:
                section_data['single_image'] = SingleImageConfig(**single_image_data)
            else:
                section_data['single_image'] = None
        
        # Convert enum strings to enums
        if 'type' in section_data and section_name == 'model':
            section_data['type'] = ModelType(section_data['type'])
            section_data['inference_mode'] = InferenceMode(section_data['inference_mode'])
        
        if 'smoothing_method' in section_data:
            section_data['smoothing_method'] = SmoothingMethod(section_data['smoothing_method'])
        
        if 'formats' in section_data:
            section_data['formats'] = [OutputFormat(f) for f in section_data['formats']]
        
        if 'level' in section_data and section_name == 'logging':
            section_data['level'] = LogLevel(section_data['level'])
        
        return dataclass_type(**section_data)
    
    def _validate_config(self):
        """Cross-section validation"""
        # Ensure single-image config exists for single-image mode
        if self.model.inference_mode == InferenceMode.SINGLE_IMAGE:
            if self.postprocessing.single_image is None:
                # Use defaults
                self.postprocessing.single_image = SingleImageConfig()
        
        # Ensure visualization output dir is set
        if self.output.save_visualizations and self.output.visualization_output_dir is None:
            self.output.visualization_output_dir = os.path.join(
                self.experiment.output_dir, 'visualizations'
            )
            Path(self.output.visualization_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Ensure log file path is set
        if self.logging.log_file is None:
            self.logging.log_file = os.path.join(
                self.experiment.output_dir, 'pipeline.log'
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary (for saving/logging)"""
        from dataclasses import asdict
        return {
            'experiment': asdict(self.experiment),
            'data': asdict(self.data),
            'preprocessing': asdict(self.preprocessing),
            'model': asdict(self.model),
            'postprocessing': asdict(self.postprocessing),
            'quality_control': asdict(self.quality_control),
            'output': asdict(self.output),
            'logging': asdict(self.logging),
        }
    
    def save(self, output_path: str):
        """Save configuration to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        return f"PipelineConfig(experiment='{self.experiment.name}', model={self.model.type.value}, mode={self.model.inference_mode.value})"


def load_config(config_path: str, **overrides) -> PipelineConfig:
    """
    Convenience function to load configuration.
    """
    return PipelineConfig(config_path, override_params=overrides if overrides else None)
