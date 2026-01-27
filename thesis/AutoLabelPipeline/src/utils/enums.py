"""
Enumerations and type definitions for the turn signal detection pipeline.
"""
from enum import Enum


class ModelType(str, Enum):
    """Model architecture types"""
    COSMOS_REASON1 = "cosmos_reason1"
    COSMOS_REASON2 = "cosmos_reason2"
    QWEN3_VL = "qwen3_vl"
    QWEN25_VL = "qwen25_vl"


class InferenceMode(str, Enum):
    """How the model processes input"""
    VIDEO = "video"          # Processes sequence of frames as video
    SINGLE_IMAGE = "single"  # Processes individual frames


class TurnSignalLabel(str, Enum):
    """Turn signal states"""
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    UNCERTAIN = "uncertain"  # For low-confidence predictions


class SmoothingMethod(str, Enum):
    """Temporal smoothing algorithms"""
    MEDIAN = "median"
    MODE = "mode"
    HMM = "hmm"
    THRESHOLD = "threshold"
    NONE = "none"


class OutputFormat(str, Enum):
    """Output file formats"""
    JSON = "json"
    CSV = "csv"
    COCO = "coco"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
