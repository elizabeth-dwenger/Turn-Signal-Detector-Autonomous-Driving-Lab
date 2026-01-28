"""
Script to download and cache all required models.
Run this once on the cluster to pre-download models.

Usage:
    Interactive: python download_models.py
    Non-interactive: python download_models.py --yes
"""
import os
import sys
import gc
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoProcessor, AutoModelForVision2Seq
import torch

# Ensure HF_HOME is set
HF_HOME = os.environ.get('HF_HOME', '/gpfs/helios/home/dwenger/models/huggingface')
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = f"{HF_HOME}/transformers"

print(f"Models will be cached to: {HF_HOME}")
print(f"Transformers cache: {os.environ['TRANSFORMERS_CACHE']}")

# Models to download
MODELS = {
    'cosmos_reason_7b': {
        'name': 'nvidia/Cosmos-Reason1-7B',
        'type': 'vlm',
        'size': '~14GB',
    },
    'cosmos_reason_8b': {
        'name': 'nvidia/Cosmos-Reason2-8B',
        'type': 'vlm',
        'size': '~16GB',
    },
    'qwen3_vl': {
        'name': 'Qwen/Qwen3-VL-8B-Instruct',
        'type': 'vlm',
        'size': '~15GB',
    },
    'qwen25_vl': {
        'name': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'type': 'vlm',
        'size': '~15GB',
    },
}


def check_disk_space():
    """Check available disk space"""
    import shutil
    
    cache_dir = Path(HF_HOME)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    total, used, free = shutil.disk_usage(cache_dir)
    
    print(f"\nDisk space at {cache_dir}:")
    print(f"  Total: {total // (2**30)} GB")
    print(f"  Used:  {used // (2**30)} GB")
    print(f"  Free:  {free // (2**30)} GB")
    
    required_gb = 80  # Estimate
    if free < required_gb * (2**30):
        print(f"\n  WARNING: Low disk space!")
        print(f"  Required: ~{required_gb} GB")
        print(f"  Available: {free // (2**30)} GB")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    return free

def download_model(model_id: str, model_info: dict):
    """Download a single model using the Vision2Seq class"""
    print("\n" + "=" * 80)
    print(f"Downloading: {model_id}")
    print(f"  Name: {model_info['name']}")
    print("=" * 80)
    
    try:
        # 1. Download Processor
        print(f"\n[1/2] Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_info['name'],
            trust_remote_code=True,
            token=os.environ.get('HF_TOKEN')
        )
        
        # 2. Download Model Weights
        print(f"\n[2/2] Downloading model weights...")
        # Use AutoModelForVision2Seq instead of AutoModelForCausalLM
        model = AutoModelForVision2Seq.from_pretrained(
            model_info['name'],
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='cpu',
            low_cpu_mem_usage=True,
            token=os.environ.get('HF_TOKEN')
        )
        
        print(f"✓ {model_id} downloaded successfully!")
        
        # Cleanup
        del model
        del processor
        gc.collect()
        return True
    
    except Exception as e:
        print(f"\n✗ Error downloading {model_id}:")
        print(f"  {type(e).__name__}: {e}")
        
        if "403" in str(e):
            print(f"\n  (!) ACTION REQUIRED: You must manually accept the license for this model at:")
            print(f"      https://huggingface.co/{model_info['name']}")
            
        return False


def verify_model(model_id: str, model_info: dict):
    """Verify model is downloaded correctly"""
    print(f"\nVerifying {model_id}...")
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_info['name'],
            trust_remote_code=True,
            local_files_only=True  # Only check local cache
        )
        print(f"Model verified in cache")
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Download VLM models for turn signal detection')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt (for non-interactive use)')
    args = parser.parse_args()
    
    print("=" * 80)
    print(" " * 20 + "MODEL DOWNLOAD SCRIPT")
    print("=" * 80)
    
    # Check disk space
    free_space = check_disk_space()
    
    # Ask for confirmation (auto-yes if non-interactive or --yes flag)
    print(f"\nModels to download: {len(MODELS)}")
    for model_id, info in MODELS.items():
        print(f"  - {info['name']} ({info['size']})")
    
    print(f"\nTotal estimated size: ~60-80 GB")
    print(f"Available space: {free_space // (2**30)} GB")
    
    # Auto-proceed if --yes flag or non-interactive
    if args.yes or not sys.stdin.isatty():
        print("\n→ Proceeding automatically (non-interactive mode)")
        print("  (To cancel a running job: scancel <job_id>)")
    else:
        response = input("\nProceed with download? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            sys.exit(0)
    
    # Download each model
    results = {}
    for model_id, model_info in MODELS.items():
        success = download_model(model_id, model_info)
        results[model_id] = success
        
        if success:
            verify_model(model_id, model_info)
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "SUMMARY")
    print("=" * 80)
    
    for model_id, success in results.items():
        status = "SUCCESS" if success else "✗ FAILED"
        print(f"  {model_id:.<50} {status}")
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\n  Downloaded: {successful}/{total} models")
    
    if successful == total:
        print("\n  All models downloaded successfully!")
        print(f"\n  Models cached at: {HF_HOME}")
        print("\n  Next steps:")
        print("  1. Set HF_HOME in your job scripts")
        print("  2. Models will load from cache (no re-download)")
        print("  3. Run test_model_loading.py to verify")
    else:
        print("\n  Some models failed to download")
        print("  Check errors above and retry failed models")
    
    return successful == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
