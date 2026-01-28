"""
Model factory for creating appropriate detector instances.
"""
import logging
from typing import Optional

from utils.enums import ModelType
from .base import TurnSignalDetector
from .qwen_vl import QwenVLDetector
from .cosmos import CosmosDetector


logger = logging.getLogger(__name__)


def create_model(model_config) -> TurnSignalDetector:
    """
    Factory function to create appropriate model instance.
    """
    model_type = model_config.type
    
    logger.info(f"Creating model: {model_type.value}")
    logger.info(f"  Model path: {model_config.model_name_or_path}")
    logger.info(f"  Inference mode: {model_config.inference_mode.value}")
    logger.info(f"  Device: {model_config.device}")
    
    # Map model types to classes
    MODEL_CLASSES = {
        ModelType.QWEN3_VL: QwenVLDetector,
        ModelType.QWEN25_VL: QwenVLDetector,  # Same class for both Qwen versions
        ModelType.COSMOS_REASON1: CosmosDetector,
        ModelType.COSMOS_REASON2: CosmosDetector,  # Same class for both Cosmos versions
    }
    
    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {list(MODEL_CLASSES.keys())}"
        )
    
    # Create model instance
    model_class = MODEL_CLASSES[model_type]
    model = model_class(model_config)
    
    logger.info(f" Created {model}")
    
    return model


def load_model(model_config, warmup: bool = True) -> TurnSignalDetector:
    """
    Create and load model in one step.
    """
    model = create_model(model_config)
    
    if warmup:
        logger.info("Warming up model...")
        model.warmup()
        logger.info(" Model ready for inference")
    
    return model
