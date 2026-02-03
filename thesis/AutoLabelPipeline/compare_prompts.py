#!/usr/bin/env python
"""
Prompt Comparison Tool - Test multiple prompts on same sequences.

Helps you:
1. Compare different prompt formulations
2. A/B test prompt changes
3. Track prompt performance over iterations
"""
import sys
import argparse
from pathlib import Path
import json
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config import load_config
from data import (
    load_dataset_from_config,
    create_image_loader,
    SequencePreprocessor
)
from models import load_model


def test_prompt(config_path: str, prompt_file: str, test_sequences_file: str,
                output_dir: str = "prompt_comparison"):
    """
    Test a single prompt on test sequences.
    """
    # Load config
    config = load_config(config_path)
    
    # Load test sequence IDs
    with open(test_sequences_file, 'r') as f:
        test_set = json.load(f)
        sequence_ids = test_set['sequence_ids']
    
    # Override config
    config.data.sequence_filter = sequence_ids
    config.data.max_sequences = None
    config.model.prompt_template_path = prompt_file
    
    print(f"\nTesting prompt: {prompt_file}")
    print(f"Test sequences: {len(sequence_ids)}")
    
    # Create timestamped output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_name = Path(prompt_file).stem
    model_name = config.model.type.value
    
    run_output_dir = Path(output_dir) / f"{model_name}_{prompt_name}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_output_dir}")
    
    # Load dataset
    dataset = load_dataset_from_config(config.data)
    print(f"Loaded {dataset.num_sequences} sequences")
    
    # Setup
    image_loader = create_image_loader(config.data, lazy=True)
    preprocessor = SequencePreprocessor(config.preprocessing)
    
    # Load model
    print("Loading model...")
    model = load_model(config.model, warmup=True)
    
    # Run inference
    results = []
    
    for sequence in tqdm(dataset.sequences, desc="Processing"):
        # Load images
        image_loader(sequence, load_full_frame=False)
        
        loaded = sum(1 for f in sequence.frames if f.crop_image is not None)
        if loaded == 0:
            continue
        
        try:
            # Predict
            if config.model.inference_mode.value == 'video':
                video = preprocessor.preprocess_for_video(sequence)
                prediction = model.predict_video(video)
                predictions = [prediction]
            else:
                samples = preprocessor.preprocess_for_single_images(sequence)
                images = [s[0] for s in samples]
                predictions = model.predict_batch(images)
            
            # Store result
            results.append({
                'sequence_id': sequence.sequence_id,
                'num_frames': sequence.num_frames,
                'ground_truth': sequence.ground_truth_label if sequence.has_ground_truth else None,
                'prediction': predictions[0] if config.model.inference_mode.value == 'video' else predictions,
                'mode': config.model.inference_mode.value
            })
        
        except Exception as e:
            logging.error(f"Error processing {sequence.sequence_id}: {e}")
        
        finally:
            if hasattr(image_loader, 'clear_sequence'):
                image_loader.clear_sequence(sequence)
    
    # Get metrics
    metrics = model.get_metrics()
    
    # Calculate accuracy
    accuracy = None
    if any(r['ground_truth'] for r in results):
        if config.model.inference_mode.value == 'video':
            correct = sum(1 for r in results
                         if r['ground_truth'] and r['prediction']['label'] == r['ground_truth'])
            total = sum(1 for r in results if r['ground_truth'])
            accuracy = correct / total if total > 0 else 0
        else:
            # Majority vote for single-image
            correct = 0
            total = 0
            for r in results:
                if r['ground_truth']:
                    from collections import Counter
                    labels = [p['label'] for p in r['prediction']]
                    most_common = Counter(labels).most_common(1)[0][0]
                    if most_common == r['ground_truth']:
                        correct += 1
                    total += 1
            accuracy = correct / total if total > 0 else 0
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save in the run-specific directory
    result_file = run_output_dir / "results.json"
    
    output = {
        'prompt_file': prompt_file,
        'test_set': test_sequences_file,
        'timestamp': timestamp,
        'model': model_name,
        'num_sequences': len(results),
        'accuracy': accuracy,
        'metrics': metrics,
        'results': results
    }
    
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Also save summary to main output_dir for easy comparison
    summary_file = output_path / f"{model_name}_{prompt_name}_{timestamp}_summary.json"
    summary = {
        'prompt_file': prompt_file,
        'timestamp': timestamp,
        'model': model_name,
        'num_sequences': len(results),
        'accuracy': accuracy,
        'metrics': metrics,
        'run_directory': str(run_output_dir)
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n Results saved to {result_file}")
    print(f" Summary saved to {summary_file}")
    print(f"\nPerformance:")
    print(f"  Accuracy: {accuracy:.1%}" if accuracy else "  Accuracy: N/A (no ground truth)")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.1f} ms")
    print(f"  Parse Success: {metrics['parse_success_rate']:.1%}")
    
    return output


def compare_prompts(comparison_dir: str):
    """
    Compare multiple prompt test results.
    """
    result_files = list(Path(comparison_dir).glob("*.json"))
    
    if not result_files:
        print(f"No result files found in {comparison_dir}")
        return
    
    print(f"\nComparing {len(result_files)} prompt results:")
    print("="*80)
    
    comparison_data = []
    
    for result_file in sorted(result_files):
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        comparison_data.append({
            'Prompt': Path(data['prompt_file']).stem,
            'Timestamp': data['timestamp'],
            'Sequences': data['num_sequences'],
            'Accuracy': f"{data['accuracy']:.1%}" if data['accuracy'] else "N/A",
            'Avg Latency (ms)': f"{data['metrics']['avg_latency_ms']:.1f}",
            'Parse Success': f"{data['metrics']['parse_success_rate']:.1%}"
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Save comparison
    comparison_file = Path(comparison_dir) / "comparison_summary.csv"
    df.to_csv(comparison_file, index=False)
    print(f"\n Comparison saved to {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare different prompts')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a prompt')
    test_parser.add_argument('--config', type=str, required=True,
                            help='Configuration file')
    test_parser.add_argument('--prompt', type=str, required=True,
                            help='Prompt file to test')
    test_parser.add_argument('--test-set', type=str, required=True,
                            help='Test sequences JSON file')
    test_parser.add_argument('--output-dir', type=str, default='prompt_comparison',
                            help='Output directory')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare prompt results')
    compare_parser.add_argument('--dir', type=str, required=True,
                               help='Directory with result files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'test':
        test_prompt(
            args.config,
            args.prompt,
            args.test_set,
            args.output_dir
        )
    
    elif args.command == 'compare':
        compare_prompts(args.dir)


if __name__ == '__main__':
    main()
