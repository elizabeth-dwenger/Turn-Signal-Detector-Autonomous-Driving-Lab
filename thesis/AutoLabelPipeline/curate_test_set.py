#!/usr/bin/env python
"""
1. Browse available sequences
2. Select diverse test cases
3. Save curated test sets
4. Analyze sequence characteristics
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import json
from collections import Counter, defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config import load_config
from data import load_dataset_from_config


class TestSetCurator:
    """Manage test sequence selection"""
    
    def __init__(self, config_path: str):
        """Load dataset for curation"""
        config = load_config(config_path)
        
        # Load full dataset (no filtering)
        config.data.max_sequences = None
        config.data.sequence_filter = None
        
        print(f"Loading dataset from: {config.data.input_csv}")
        self.dataset = load_dataset_from_config(config.data)
        print(f"Loaded {self.dataset.num_sequences} sequences")
        
        # Analyze sequences
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Analyze sequence characteristics"""
        self.stats = {
            'total_sequences': self.dataset.num_sequences,
            'total_frames': self.dataset.total_frames,
            'length_distribution': defaultdict(int),
            'label_distribution': defaultdict(int),
            'video_distribution': defaultdict(int),
            'camera_distribution': defaultdict(int),
        }
        
        for seq in self.dataset.sequences:
            # Length buckets
            length_bucket = (seq.num_frames // 10) * 10
            self.stats['length_distribution'][f"{length_bucket}-{length_bucket+9}"] += 1
            
            # Labels
            if seq.has_ground_truth:
                label = seq.ground_truth_label
                self.stats['label_distribution'][label] += 1
            
            # Video/Camera
            self.stats['video_distribution'][seq.video_name] += 1
            self.stats['camera_distribution'][seq.camera_name] += 1
    
    def show_statistics(self):
        """Display dataset statistics"""
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        
        print(f"\nTotal Sequences: {self.stats['total_sequences']}")
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Avg Frames/Sequence: {self.stats['total_frames']/self.stats['total_sequences']:.1f}")
        
        print(f"\nSequence Length Distribution:")
        for length_range, count in sorted(self.stats['length_distribution'].items()):
            pct = count / self.stats['total_sequences'] * 100
            print(f"  {length_range:12s}: {count:4d} ({pct:5.1f}%)")
        
        if self.stats['label_distribution']:
            print(f"\nGround Truth Label Distribution:")
            for label, count in sorted(self.stats['label_distribution'].items()):
                pct = count / sum(self.stats['label_distribution'].values()) * 100
                print(f"  {label:8s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\nTop 10 Videos:")
        for video, count in sorted(self.stats['video_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {video[:60]:60s}: {count:3d} sequences")
        
        print(f"\nCamera Distribution:")
        for camera, count in sorted(self.stats['camera_distribution'].items()):
            pct = count / self.stats['total_sequences'] * 100
            print(f"  {camera:20s}: {count:4d} ({pct:5.1f}%)")
    
    def browse_sequences(self, num_show: int = 20, filter_label: str = None,
                        min_frames: int = None, max_frames: int = None):
        """Browse sequences with filters"""
        
        filtered = self.dataset.sequences
        
        # Apply filters
        if filter_label:
            filtered = [s for s in filtered if s.ground_truth_label == filter_label]
        
        if min_frames:
            filtered = [s for s in filtered if s.num_frames >= min_frames]
        
        if max_frames:
            filtered = [s for s in filtered if s.num_frames <= max_frames]
        
        print(f"\nShowing {min(num_show, len(filtered))}/{len(filtered)} sequences:")
        print("-"*80)
        
        for i, seq in enumerate(filtered[:num_show]):
            gt = seq.ground_truth_label if seq.has_ground_truth else "N/A"
            print(f"{i+1:3d}. {seq.sequence_id[:65]:65s} | {seq.num_frames:3d}f | GT: {gt}")
        
        if len(filtered) > num_show:
            print(f"... and {len(filtered) - num_show} more")
        
        return filtered
    
    def create_stratified_test_set(self, num_per_label: int = 5, 
                                  output_file: str = "test_sequences.json"):
        """Create balanced test set with equal samples per label"""
        
        # Group by label
        by_label = defaultdict(list)
        for seq in self.dataset.sequences:
            if seq.has_ground_truth:
                by_label[seq.ground_truth_label].append(seq)
        
        if not by_label:
            print("No sequences with ground truth labels found!")
            return None
        
        # Sample from each label
        test_set = []
        for label, sequences in sorted(by_label.items()):
            num_available = len(sequences)
            num_to_sample = min(num_per_label, num_available)
            
            sampled = random.sample(sequences, num_to_sample)
            test_set.extend(sampled)
            
            print(f"Label '{label}': sampled {num_to_sample}/{num_available} sequences")
        
        # Save
        test_sequence_ids = [s.sequence_id for s in test_set]
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'description': f'Stratified test set: {num_per_label} per label',
                'num_sequences': len(test_sequence_ids),
                'sequence_ids': test_sequence_ids
            }, f, indent=2)
        
        print(f"\n Saved {len(test_sequence_ids)} test sequences to {output_path}")
        
        return test_sequence_ids
    
    def create_diverse_test_set(self, num_sequences: int = 20,
                               output_file: str = "test_sequences.json"):
        """Create diverse test set across videos, cameras, and lengths"""
        
        # Diversity criteria:
        # 1. Different videos
        # 2. Different cameras
        # 3. Different sequence lengths
        # 4. Different labels (if available)
        
        videos_used = set()
        cameras_used = set()
        labels_used = defaultdict(int)
        
        candidates = list(self.dataset.sequences)
        random.shuffle(candidates)
        
        selected = []
        
        for seq in candidates:
            if len(selected) >= num_sequences:
                break
            
            # Diversity score (higher = more diverse)
            score = 0
            
            if seq.video_name not in videos_used:
                score += 3
            if seq.camera_name not in cameras_used:
                score += 2
            
            if seq.has_ground_truth:
                if labels_used[seq.ground_truth_label] == 0:
                    score += 5
                elif labels_used[seq.ground_truth_label] < 3:
                    score += 2
            
            # Prefer medium-length sequences
            if 10 <= seq.num_frames <= 50:
                score += 1
            
            if score > 0 or len(selected) < 5:  # Always add first few
                selected.append(seq)
                videos_used.add(seq.video_name)
                cameras_used.add(seq.camera_name)
                if seq.has_ground_truth:
                    labels_used[seq.ground_truth_label] += 1
        
        # Save
        test_sequence_ids = [s.sequence_id for s in selected]
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'description': f'Diverse test set: {num_sequences} sequences',
                'num_sequences': len(test_sequence_ids),
                'videos_covered': len(videos_used),
                'cameras_covered': len(cameras_used),
                'label_distribution': dict(labels_used),
                'sequence_ids': test_sequence_ids
            }, f, indent=2)
        
        print(f"\n Saved {len(test_sequence_ids)} diverse test sequences to {output_path}")
        print(f"  Videos covered: {len(videos_used)}")
        print(f"  Cameras covered: {len(cameras_used)}")
        print(f"  Label distribution: {dict(labels_used)}")
        
        return test_sequence_ids
    
    def create_custom_test_set(self, sequence_ids: list, 
                              output_file: str = "test_sequences.json"):
        """Save custom list of sequence IDs"""
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'description': 'Custom test set',
                'num_sequences': len(sequence_ids),
                'sequence_ids': sequence_ids
            }, f, indent=2)
        
        print(f"\n Saved {len(sequence_ids)} custom test sequences to {output_path}")
        
        return sequence_ids
    
    def export_sequence_list(self, output_file: str = "all_sequences.csv"):
        """Export full sequence list to CSV for manual selection"""
        
        data = []
        for seq in self.dataset.sequences:
            data.append({
                'sequence_id': seq.sequence_id,
                'num_frames': seq.num_frames,
                'video': seq.video_name,
                'camera': seq.camera_name,
                'ground_truth': seq.ground_truth_label if seq.has_ground_truth else None
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"\n Exported {len(data)} sequences to {output_file}")
        print(f"  You can manually edit this file and load selected sequences")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Curate test sequences for prompt development')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Stats command
    subparsers.add_parser('stats', help='Show dataset statistics')
    
    # Browse command
    browse_parser = subparsers.add_parser('browse', help='Browse sequences')
    browse_parser.add_argument('--num', type=int, default=20,
                              help='Number of sequences to show')
    browse_parser.add_argument('--label', type=str, help='Filter by ground truth label')
    browse_parser.add_argument('--min-frames', type=int, help='Minimum number of frames')
    browse_parser.add_argument('--max-frames', type=int, help='Maximum number of frames')
    
    # Create stratified test set
    stratified_parser = subparsers.add_parser('stratified', 
                                             help='Create stratified test set')
    stratified_parser.add_argument('--num-per-label', type=int, default=5,
                                  help='Sequences per label')
    stratified_parser.add_argument('--output', type=str, default='test_sets/stratified.json',
                                  help='Output file')
    
    # Create diverse test set
    diverse_parser = subparsers.add_parser('diverse', 
                                          help='Create diverse test set')
    diverse_parser.add_argument('--num', type=int, default=20,
                               help='Number of sequences')
    diverse_parser.add_argument('--output', type=str, default='test_sets/diverse.json',
                               help='Output file')
    
    # Export all
    export_parser = subparsers.add_parser('export', 
                                         help='Export all sequences to CSV')
    export_parser.add_argument('--output', type=str, default='all_sequences.csv',
                              help='Output CSV file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load curator
    curator = TestSetCurator(args.config)
    
    # Execute command
    if args.command == 'stats':
        curator.show_statistics()
    
    elif args.command == 'browse':
        curator.browse_sequences(
            num_show=args.num,
            filter_label=args.label,
            min_frames=args.min_frames,
            max_frames=args.max_frames
        )
    
    elif args.command == 'stratified':
        curator.create_stratified_test_set(
            num_per_label=args.num_per_label,
            output_file=args.output
        )
    
    elif args.command == 'diverse':
        curator.create_diverse_test_set(
            num_sequences=args.num,
            output_file=args.output
        )
    
    elif args.command == 'export':
        curator.export_sequence_list(output_file=args.output)


if __name__ == '__main__':
    main()
