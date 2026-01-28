import os
import sys
from pathlib import Path
import torch
import gc
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq

HF_HOME = os.environ.get('HF_HOME', '/gpfs/helios/home/dwenger/models/huggingface')
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = f"{HF_HOME}/transformers"

MODELS = [
    'nvidia/Cosmos-Reason1-7B',
    'nvidia/Cosmos-Reason2-8B',
    'Qwen/Qwen3-VL-8B-Instruct',
    'Qwen/Qwen2.5-VL-7B-Instruct',
]

def test_model_config(model_name: str) -> bool:
    print(f"\n{'='*80}\nTesting: {model_name}\n{'='*80}")
    
    try:
        print("  [1/3] Loading config...")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True
        )
        print(f"    Config loaded (Type: {config.model_type})")
        
        print("\n  [2/3] Loading processor...")
        # VLMs need AutoProcessor, not just AutoTokenizer
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True
        )
        print(f"    Processor loaded")
        
        print("\n  [3/3] Verifying model weights on disk...")
        from transformers.utils import cached_file
        # This checks if the heaviest file (config.json) exists in cache
        model_file = cached_file(model_name, "config.json", local_files_only=True)
        print(f"    Cache verified at: {Path(model_file).parent.name}")
        
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_model_inference(model_name: str) -> bool:
    """Optional load to RAM check"""
    print(f"\n  [OPTIONAL] Testing RAM loading...")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='cpu',
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        print(f"    Model loaded to RAM successfully")
        del model
        gc.collect()
        return True
    except Exception as e:
        print(f"    Load failed: {e}")
        return False

def check_cache_size():
    """Display cache directory size"""
    import subprocess
    
    cache_dir = Path(HF_HOME)
    if not cache_dir.exists():
        print(f"\n Cache directory does not exist: {cache_dir}")
        return
    
    try:
        result = subprocess.run(
            ['du', '-sh', str(cache_dir)],
            capture_output=True,
            text=True
        )
        size = result.stdout.split()[0]
        print(f"\nCache size: {size}")
    except:
        print("\nCould not determine cache size")


def main():
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    # Check cache
    check_cache_size()
    
    # Test each model
    results = {}
    for model_name in MODELS:
        success = test_model_config(model_name)
        results[model_name] = success
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "SUMMARY")
    print("=" * 80)
    
    for model_name, success in results.items():
        status = "READY" if success else "NOT FOUND"
        short_name = model_name.split('/')[-1]
        print(f"  {short_name:.<60} {status}")
    
    ready_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n  Models ready: {ready_count}/{total_count}")
    
    if ready_count == total_count:
        print("\n  All models are accessible!")
        print("\n  Next steps:")
        print("  1. Models will load from cache (fast)")
        print("  2. Set HF_HOME in your job scripts")
    else:
        print("\n  Some models are missing")
        print("  Run: sbatch download_models.sh")
    
    return ready_count == total_count


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
