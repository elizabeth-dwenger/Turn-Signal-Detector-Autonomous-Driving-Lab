import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config import load_config, PipelineConfig
from utils.enums import InferenceMode, ModelType


def test_config_loading():
    """Test loading different configuration files"""
    
    configs_to_test = [
        'configs/cosmos_reason1_video.yaml',
        'configs/qwen3_vl_video.yaml',
        'configs/qwen3_vl_single.yaml',
    ]
    
    print("=" * 80)
    print("CONFIGURATION SYSTEM TEST")
    print("=" * 80)
    
    for config_path in configs_to_test:
        print(f"\n{'=' * 80}")
        print(f"Loading: {config_path}")
        print('=' * 80)
        
        try:
            config = load_config(config_path)
            print(f"Successfully loaded: {config}")
            
            # Display key settings
            print(f"\n  Experiment:")
            print(f"    Name: {config.experiment.name}")
            print(f"    Output: {config.experiment.output_dir}")
            
            print(f"\n  Model:")
            print(f"    Type: {config.model.type.value}")
            print(f"    Mode: {config.model.inference_mode.value}")
            print(f"    Path: {config.model.model_name_or_path}")
            print(f"    Device: {config.model.device}")
            print(f"    Batch size: {config.model.batch_size}")
            
            print(f"\n  Data:")
            print(f"    Input CSV: {config.data.input_csv}")
            print(f"    Crop dir: {config.data.crop_base_dir}")
            
            print(f"\n  Preprocessing:")
            print(f"    Resolution: {config.preprocessing.resize_resolution}")
            print(f"    Max sequence length: {config.preprocessing.max_sequence_length}")
            
            print(f"\n  Postprocessing:")
            print(f"    Smoothing: {config.postprocessing.smoothing_method.value}")
            print(f"    Window size: {config.postprocessing.smoothing_window_size}")
            print(f"    Confidence threshold: {config.postprocessing.confidence_threshold}")
            
            # Check single-image specific config
            if config.model.inference_mode == InferenceMode.SINGLE_IMAGE:
                si_config = config.postprocessing.single_image
                print(f"\n  Single-Image Mode:")
                print(f"    Min signal duration: {si_config.min_signal_duration_frames}")
                print(f"    Max gap: {si_config.max_gap_frames}")
                print(f"    Start threshold: {si_config.confidence_threshold_start}")
                print(f"    Continue threshold: {si_config.confidence_threshold_continue}")
            
            print(f"\n  Output:")
            print(f"    Formats: {[f.value for f in config.output.formats]}")
            print(f"    Visualizations: {config.output.save_visualizations}")
            
            # Test saving config
            output_path = Path(config.experiment.output_dir) / 'config_snapshot.yaml'
            config.save(str(output_path))
            print(f"\n  Saved config snapshot to: {output_path}")
            
        except Exception as e:
            print(f"Error loading {config_path}:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


def test_config_overrides():
    """Test runtime parameter overrides"""
    
    print(f"\n{'=' * 80}")
    print("TESTING CONFIGURATION OVERRIDES")
    print('=' * 80)
    
    config_path = 'configs/qwen3_vl_video.yaml'
    
    # Load with overrides
    config = load_config(
        config_path,
        experiment={
            'name': 'qwen3_vl_video_TEST_RUN',
            'random_seed': 123
        },
        model={
            'batch_size': 2,
            'temperature': 0.1
        }
    )
    
    print(f"Loaded config with overrides")
    print(f"  Experiment name: {config.experiment.name}")
    print(f"  Random seed: {config.experiment.random_seed}")
    print(f"  Batch size: {config.model.batch_size}")
    print(f"  Temperature: {config.model.temperature}")


def test_enum_validation():
    """Test that enum validation works"""
    
    print(f"\n{'=' * 80}")
    print("TESTING ENUM VALIDATION")
    print('=' * 80)
    
    # This should work
    try:
        from utils.enums import ModelType
        model_type = ModelType("qwen3_vl")
        print(f"Valid model type: {model_type.value}")
    except ValueError as e:
        print(f"Validation failed: {e}")
    
    # This should fail
    try:
        from utils.enums import ModelType
        invalid_type = ModelType("invalid_model")
        print(f"Should have failed but got: {invalid_type}")
    except ValueError as e:
        print(f"Correctly rejected invalid model type: {e}")


def demo_config_dict_conversion():
    """Demonstrate converting config to/from dict"""
    
    print(f"\n{'=' * 80}")
    print("CONFIG DICT CONVERSION")
    print('=' * 80)
    
    config = load_config('configs/qwen3_vl_single.yaml')
    config_dict = config.to_dict()
    
    print(f"Converted config to dictionary")
    print(f"  Keys: {list(config_dict.keys())}")
    
    # Example: access nested values
    print(f"\n  Example nested access:")
    print(f"    Model type: {config_dict['model']['type']}")
    print(f"    Smoothing method: {config_dict['postprocessing']['smoothing_method']}")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print(" " * 20 + "PIPELINE CONFIGURATION TEST SUITE")
    print("=" * 80 + "\n")
    
    test_config_loading()
    test_config_overrides()
    test_enum_validation()
    demo_config_dict_conversion()
    
    print("\n" + "=" * 80)
    print(" " * 30 + "ALL TESTS COMPLETED")
    print("=" * 80 + "\n")
