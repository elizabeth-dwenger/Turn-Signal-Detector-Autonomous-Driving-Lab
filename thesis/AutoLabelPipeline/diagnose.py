#!/usr/bin/env python
"""
Diagnostic tool for debugging turn signal detection issues.

Usage:
    python diagnose.py --result-dir results/cosmos_reason1_video/test_runs/cosmos_reason1_20260202_120724
    python diagnose.py --sequence-file results/.../sequences/seq_id.json
"""
import sys
import argparse
import json
from pathlib import Path
from collections import Counter
import pandas as pd


def diagnose_result_directory(result_dir):
    """Analyze all results in a directory"""
    result_path = Path(result_dir)
    
    print("="*80)
    print(f"DIAGNOSING: {result_dir}")
    print("="*80)
    
    # Find all sequence JSON files
    if (result_path / "sequences").exists():
        seq_dir = result_path / "sequences"
    else:
        seq_dir = result_path
    
    json_files = list(seq_dir.glob("*.json"))
    
    if not json_files:
        print(f"\n❌ No JSON files found in {seq_dir}")
        return
    
    print(f"\n✓ Found {len(json_files)} sequence files")
    
    # Analyze each sequence
    issues = {
        'truncated_output': [],
        'missing_temporal': [],
        'parse_failures': [],
        'empty_predictions': []
    }
    
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        
        seq_id = data.get('sequence_id', json_file.stem)
        predictions = data.get('predictions', [])
        
        if not predictions:
            issues['empty_predictions'].append(seq_id)
            continue
        
        for pred in predictions:
            # Check for truncated output
            raw = pred.get('raw_output', '')
            if raw and not raw.strip().endswith('}'):
                if seq_id not in issues['truncated_output']:
                    issues['truncated_output'].append(seq_id)
            
            # Check for missing temporal info
            if pred.get('start_frame') is None and pred.get('label') != 'none':
                if seq_id not in issues['missing_temporal']:
                    issues['missing_temporal'].append(seq_id)
            
            # Check for parse failures
            if 'parse failed' in pred.get('reasoning', '').lower():
                if seq_id not in issues['parse_failures']:
                    issues['parse_failures'].append(seq_id)
    
    # Print summary
    print(f"\n" + "="*80)
    print("ISSUE SUMMARY")
    print("="*80)
    
    total_issues = sum(len(v) for v in issues.values())
    
    if total_issues == 0:
        print("\n✓ No issues found! All predictions look good.")
    else:
        print(f"\n⚠️  Found {total_issues} sequences with issues:\n")
        
        if issues['empty_predictions']:
            print(f"  Empty predictions: {len(issues['empty_predictions'])}")
            for seq in issues['empty_predictions'][:3]:
                print(f"    - {seq}")
        
        if issues['truncated_output']:
            print(f"\n  Truncated model output: {len(issues['truncated_output'])}")
            for seq in issues['truncated_output'][:3]:
                print(f"    - {seq}")
        
        if issues['missing_temporal']:
            print(f"\n  Missing temporal info: {len(issues['missing_temporal'])}")
            for seq in issues['missing_temporal'][:3]:
                print(f"    - {seq}")
        
        if issues['parse_failures']:
            print(f"\n  Parse failures: {len(issues['parse_failures'])}")
            for seq in issues['parse_failures'][:3]:
                print(f"    - {seq}")
        
    # Analyze prediction distribution
    print(f"\n" + "="*80)
    print("PREDICTION STATISTICS")
    print("="*80)
    
    all_labels = []
    has_temporal = 0
    
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        
        for pred in data.get('predictions', []):
            all_labels.append(pred.get('label', 'unknown'))
            if pred.get('start_frame') is not None:
                has_temporal += 1
    
    label_counts = Counter(all_labels)
    
    print(f"\nLabel distribution:")
    for label, count in label_counts.most_common():
        pct = count / len(all_labels) * 100 if all_labels else 0
        print(f"  {label:10s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nTemporal localization:")
    print(f"  Predictions with temporal info: {has_temporal}/{len(all_labels)} ({has_temporal/len(all_labels)*100 if all_labels else 0:.1f}%)")
    
    return issues


def diagnose_single_sequence(json_file):
    """Deep dive into a single sequence"""
    print("="*80)
    print(f"DETAILED DIAGNOSIS: {Path(json_file).name}")
    print("="*80)
    
    with open(json_file) as f:
        data = json.load(f)
    
    seq_id = data.get('sequence_id', 'unknown')
    num_frames = data.get('num_frames', 0)
    predictions = data.get('predictions', [])
    
    print(f"\nSequence ID: {seq_id}")
    print(f"Number of frames: {num_frames}")
    print(f"Number of predictions: {len(predictions)}")
    
    if not predictions:
        print("\n❌ No predictions found!")
        return
    
    # Analyze each prediction
    for i, pred in enumerate(predictions):
        print(f"\n" + "-"*80)
        print(f"Prediction {i+1}/{len(predictions)}")
        print("-"*80)
        
        # Basic info
        print(f"\nLabel: {pred.get('label', 'N/A')}")
        
        # Temporal info
        start_f = pred.get('start_frame')
        end_f = pred.get('end_frame')
        start_t = pred.get('start_time_seconds')
        end_t = pred.get('end_time_seconds')
        
        if start_f is not None or end_f is not None:
            print(f"\nTemporal localization:")
            print(f"  Start: frame {start_f}, time {start_t}s")
            print(f"  End: frame {end_f}, time {end_t}s")
        else:
            print(f"\n⚠️  No temporal localization info")
        
        # Reasoning
        reasoning = pred.get('reasoning', '')
        if reasoning:
            print(f"\nReasoning: {reasoning[:200]}")
            if len(reasoning) > 200:
                print(f"  ... (truncated, full length: {len(reasoning)} chars)")
        else:
            print(f"\n⚠️  No reasoning provided")
        
        # Raw output analysis
        raw = pred.get('raw_output', '')
        if raw:
            print(f"\nRaw output analysis:")
            print(f"  Length: {len(raw)} characters")
            
            # Check if output is complete
            if '<think>' in raw and '</think>' not in raw:
                print(f"  ⚠️  Incomplete thinking tags (missing </think>)")
            
            if '<answer>' in raw and '</answer>' not in raw:
                print(f"  ⚠️  Incomplete answer tags (missing </answer>)")
            
            # Check for JSON
            if '{' in raw:
                json_start = raw.find('{')
                json_end = raw.rfind('}')
                if json_end > json_start:
                    json_str = raw[json_start:json_end+1]
                    try:
                        parsed_json = json.loads(json_str)
                        print(f"  ✓ Valid JSON found")
                        print(f"    Keys: {list(parsed_json.keys())}")
                    except json.JSONDecodeError as e:
                        print(f"  ❌ Invalid JSON: {e}")
                else:
                    print(f"  ❌ No closing brace found for JSON")
            else:
                print(f"  ⚠️  No JSON found in output")
            
            # Show raw output sample
            print(f"\nRaw output (first 300 chars):")
            print("-" * 60)
            print(raw[:300])
            if len(raw) > 300:
                print(f"\n  ... ({len(raw) - 300} more characters)")
            print("-" * 60)
            
            print(f"\nRaw output (last 300 chars):")
            print("-" * 60)
            print(raw[-300:])
            print("-" * 60)
        else:
            print(f"\n⚠️  No raw output saved")
        
        # Check for processing flags
        if pred.get('original_label'):
            print(f"\n⚠️  Label was modified by post-processing:")
            print(f"  Original: {pred.get('original_label')}")
            print(f"  Final: {pred.get('label')}")
        
        if pred.get('constraint_enforced'):
            print(f"\n⚠️  Constraints were enforced (label may have been changed)")


def check_prompt_file(prompt_path):
    """Check if prompt file exists and show its content"""
    print("="*80)
    print(f"PROMPT FILE CHECK: {prompt_path}")
    print("="*80)
    
    if not Path(prompt_path).exists():
        print(f"\n❌ Prompt file not found: {prompt_path}")
        return
    
    with open(prompt_path) as f:
        prompt = f.read()
    
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Prompt file: {prompt_path}")
    
    # Check for key elements
    print(f"\nPrompt analysis:")
    
    checks = {
        'Asks for JSON output': '"label"' in prompt.lower() or 'json' in prompt.lower(),
        'Asks for start_frame': 'start_frame' in prompt,
        'Asks for end_frame': 'end_frame' in prompt,
        'Asks for start_time': 'start_time' in prompt,
        'Asks for end_time': 'end_time' in prompt,
        'Provides examples': 'example' in prompt.lower(),
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "❌"
        print(f"  {status} {check}")
    
    print(f"\nPrompt preview (first 500 chars):")
    print("-" * 60)
    print(prompt[:500])
    print("-" * 60)


def check_model_config(config_path):
    """Check model configuration"""
    print("="*80)
    print(f"MODEL CONFIG CHECK: {config_path}")
    print("="*80)
    
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        model_config = config.get('model', {})
        
        print(f"\nModel settings:")
        print(f"  Type: {model_config.get('type')}")
        print(f"  Mode: {model_config.get('inference_mode')}")
        print(f"  Model path: {model_config.get('model_name_or_path')}")
        print(f"  Max new tokens: {model_config.get('max_new_tokens')}")
        print(f"  Temperature: {model_config.get('temperature')}")
        print(f"  Prompt template: {model_config.get('prompt_template_path')}")
        
        # Check if max_new_tokens might be too small
        max_tokens = model_config.get('max_new_tokens', 0)
        if max_tokens < 200:
            print(f"\n⚠️  max_new_tokens={max_tokens} might be too small for detailed responses")
            print(f"    Consider increasing to 300-500 for temporal localization")
        
    except Exception as e:
        print(f"\n❌ Error reading config: {e}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose turn signal detection issues')
    parser.add_argument('--result-dir', type=str,
                       help='Directory containing test results')
    parser.add_argument('--sequence-file', type=str,
                       help='Specific sequence JSON file to analyze')
    parser.add_argument('--prompt', type=str,
                       help='Prompt file to check')
    parser.add_argument('--config', type=str,
                       help='Config file to check')
    
    args = parser.parse_args()
    
    if not any([args.result_dir, args.sequence_file, args.prompt, args.config]):
        parser.print_help()
        print("\n" + "="*80)
        print("QUICK EXAMPLES:")
        print("="*80)
        print("\n# Diagnose entire test run:")
        print("python diagnose.py --result-dir results/cosmos_reason1_video/test_runs/cosmos_reason1_20260202_120724")
        print("\n# Analyze specific sequence:")
        print("python diagnose.py --sequence-file results/.../sequences/seq_id.json")
        print("\n# Check prompt file:")
        print("python diagnose.py --prompt data/prompts/turn_signal_video_enhanced.txt")
        print("\n# Check config:")
        print("python diagnose.py --config configs/cosmos_reason1_video.yaml")
        return
    
    if args.result_dir:
        diagnose_result_directory(args.result_dir)
    
    if args.sequence_file:
        diagnose_single_sequence(args.sequence_file)
    
    if args.prompt:
        check_prompt_file(args.prompt)
    
    if args.config:
        check_model_config(args.config)


if __name__ == '__main__':
    main()
