import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json

class TurnSignalSequenceSampler:
    """
    Sample frame sequences around known turn signal instances and generate
    labeling schemas for temporal ML models.
    """
    
    def __init__(self,
                 tracking_csv: str,
                 fps: int = 30,
                 sample_interval: int = 15,  # frames between samples
                 window_seconds: float = 5.0):  # seconds before/after anchor
        self.df = pd.read_csv(tracking_csv)
        self.fps = fps
        self.sample_interval = sample_interval
        self.window_frames = int(window_seconds * fps)
        
        # Parse sequence name from crop_path if not already a column
        if 'sequence' not in self.df.columns:
            self.df['sequence'] = self.df['crop_path'].apply(
                lambda x: str(Path(x).parts[-4]) if len(Path(x).parts) >= 4 else 'unknown'
            )
    
    def find_anchor_frames(self, anchor_crop_paths: List[str]) -> pd.DataFrame:
        anchors = []
        for crop_path in anchor_crop_paths:
            match = self.df[self.df['crop_path'] == crop_path]
            if len(match) > 0:
                anchors.append(match.iloc[0])
            else:
                print(f"Warning: Crop path not found: {crop_path}")
        
        return pd.DataFrame(anchors)
    
    def sample_sequence_around_anchor(self,
                                     sequence: str,
                                     track_id: int,
                                     anchor_frame: int) -> pd.DataFrame:
        # Define frame range
        start_frame = max(0, anchor_frame - self.window_frames)
        end_frame = anchor_frame + self.window_frames
        
        # Get all frames for this track in the sequence
        track_frames = self.df[
            (self.df['sequence'] == sequence) &
            (self.df['track_id'] == track_id) &
            (self.df['frame_id'] >= start_frame) &
            (self.df['frame_id'] <= end_frame)
        ].sort_values('frame_id')
        
        # Sample at intervals
        sampled_indices = np.arange(0, len(track_frames), self.sample_interval)
        sampled = track_frames.iloc[sampled_indices].copy()
        
        return sampled
    
    def generate_sequences(self,
                          anchor_annotations: List[Dict[str, any]]) -> Tuple[pd.DataFrame, List[Dict]]:
        all_sampled = []
        metadata = []
        
        for idx, anno in enumerate(anchor_annotations):
            crop_path = anno['crop_path']
            label = anno['label']
            
            # Find anchor frame
            anchor_df = self.find_anchor_frames([crop_path])
            
            if len(anchor_df) == 0:
                continue
                
            anchor = anchor_df.iloc[0]
            
            # Sample sequence
            sampled = self.sample_sequence_around_anchor(
                sequence=anchor['sequence'],
                track_id=anchor['track_id'],
                anchor_frame=anchor['frame_id']
            )
            
            if len(sampled) == 0:
                continue
            
            # Add sequence ID
            seq_id = f"seq_{idx:04d}"
            sampled['sequence_id'] = seq_id
            sampled['label'] = label
            sampled['anchor_frame'] = anchor['frame_id']
            
            all_sampled.append(sampled)
            
            # Store metadata
            metadata.append({
                'sequence_id': seq_id,
                'label': label,
                'track_id': int(anchor['track_id']),
                'video_sequence': anchor['sequence'],
                'anchor_frame': int(anchor['frame_id']),
                'start_frame': int(sampled['frame_id'].min()),
                'end_frame': int(sampled['frame_id'].max()),
                'num_frames': len(sampled),
                'duration_seconds': (sampled['frame_id'].max() - sampled['frame_id'].min()) / self.fps
            })
        
        if len(all_sampled) == 0:
            return pd.DataFrame(), []
            
        return pd.concat(all_sampled, ignore_index=True), metadata
    
    def generate_download_list(self, sampled_df: pd.DataFrame, output_path: str):
        # Get unique paths for both crops and full images
        crop_paths = sampled_df['crop_path'].unique().tolist()
        img_paths = sampled_df['img_path'].unique().tolist()
        
        download_list = {
            'crop_paths': crop_paths,
            'img_paths': img_paths,
            'total_crops': len(crop_paths),
            'total_images': len(img_paths)
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(download_list, f, indent=2)
        
        # Also save as simple text files for easy rsync/scp
        with open(output_path.replace('.json', '_crops.txt'), 'w') as f:
            f.write('\n'.join(crop_paths))
        
        with open(output_path.replace('.json', '_images.txt'), 'w') as f:
            f.write('\n'.join(img_paths))
        
        print(f"Download lists saved to {output_path}")
        print(f"  - {len(crop_paths)} crop images")
        print(f"  - {len(img_paths)} full images")
        
        return download_list
    
    def create_3dcnn_labels(self, metadata: List[Dict], output_path: str):
        """
        Create labels for 3D CNN (e.g., C3D, I3D, SlowFast).
        Format: One video clip = one label (sequence-level classification)
        
        Schema:
        sequence_id, video_path, label, start_frame, end_frame, num_frames
        
        Example output:
        sequence_id,video_path,label,start_frame,end_frame,num_frames,label_id
        seq_0000,2024-03-25-15-40-16_mapping_tartu/camera_fl,left,120,270,10,1
        seq_0001,2024-03-25-15-40-16_mapping_tartu/camera_fl,right,450,600,10,2
        seq_0002,2024-03-25-15-40-16_mapping_tartu/camera_fl,hazard,800,950,10,3
        """
        df = pd.DataFrame(metadata)
        
        # For 3D CNN, we need to specify the video clip location
        df_out = df[['sequence_id', 'video_sequence', 'label', 'start_frame',
                     'end_frame', 'num_frames']].copy()
        df_out.columns = ['sequence_id', 'video_path', 'label', 'start_frame',
                          'end_frame', 'num_frames']
        
        # Add label encoding for classification
        label_map = {'none': 0, 'left': 1, 'right': 2, 'hazard': 3}
        df_out['label_id'] = df_out['label'].map(label_map)
        
        df_out.to_csv(output_path, index=False)
        print(f"3D CNN labels saved to {output_path}")
        print(f"  - {len(df_out)} sequences")
        print(f"  - Label distribution:\n{df_out['label'].value_counts()}")
        
        return df_out
    
    def create_lstm_labels(self, sampled_df: pd.DataFrame, output_path: str):
        """
        Create labels for LSTM/GRU models.
        Format: Sequence of frames with frame-level or sequence-level labels
        
        Schema:
        sequence_id, frame_idx, frame_id, crop_path, label
        """
        # Group by sequence_id
        sequences = []
        
        for seq_id, group in sampled_df.groupby('sequence_id'):
            group = group.sort_values('frame_id')
            
            # Add frame index within sequence
            group['frame_idx'] = range(len(group))
            
            sequences.append(group[['sequence_id', 'frame_idx', 'frame_id',
                                   'crop_path', 'label']])
        
        df_out = pd.concat(sequences, ignore_index=True)
        df_out.to_csv(output_path, index=False)
        
        print(f"LSTM labels saved to {output_path}")
        print(f"  - {df_out['sequence_id'].nunique()} sequences")
        print(f"  - {len(df_out)} total frames")
        
        return df_out
    
    def create_timesformer_labels(self, sampled_df: pd.DataFrame, output_path: str):
        """
        Create labels for TimeSformer (video transformer).
        Format: Similar to 3D CNN but with frame paths listed
        
        Schema:
        sequence_id, label, frame_paths (JSON list), num_frames
        """
        sequences = []
        
        for seq_id, group in sampled_df.groupby('sequence_id'):
            group = group.sort_values('frame_id')
            
            frame_paths = group['crop_path'].tolist()
            label = group['label'].iloc[0]
            
            sequences.append({
                'sequence_id': seq_id,
                'label': label,
                'frame_paths': json.dumps(frame_paths),  # Store as JSON string
                'num_frames': len(frame_paths),
                'start_frame': int(group['frame_id'].min()),
                'end_frame': int(group['frame_id'].max())
            })
        
        df_out = pd.DataFrame(sequences)
        
        # Add label encoding
        label_map = {'none': 0, 'left': 1, 'right': 2, 'hazard': 3}
        df_out['label_id'] = df_out['label'].map(label_map)
        
        df_out.to_csv(output_path, index=False)
        
        print(f"TimeSformer labels saved to {output_path}")
        print(f"  - {len(df_out)} sequences")
        print(f"  - Average frames per sequence: {df_out['num_frames'].mean():.1f}")
        
        return df_out
    
    def create_all_labels(self, sampled_df: pd.DataFrame, metadata: List[Dict],
                         output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all formats
        self.create_3dcnn_labels(metadata, str(output_dir / 'labels_3dcnn.csv'))
        self.create_lstm_labels(sampled_df, str(output_dir / 'labels_lstm.csv'))
        self.create_timesformer_labels(sampled_df, str(output_dir / 'labels_timesformer.csv'))
        
        # Save metadata
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nAll labels saved to {output_dir}/")


if __name__ == "__main__":
    # Initialize sampler
    sampler = TurnSignalSequenceSampler(
        tracking_csv='../tracks_deepsort.csv',
        fps=10,
        sample_interval=4,  # Sample every 2 frames
        window_seconds=2.5   # +/-2.5 seconds around anchor (5 sec total)
    )
    
    # Define your manually annotated anchor frames
    anchor_annotations = [
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-08-16-16-03-30_mapping_tartu_streets/camera_narrow_front/predict/crops/car/142145.jpg',
            'label': 'right'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-08-16-14-33-25_mapping_tartu_streets/camera_narrow_front/predict/crops/car/087908.jpg',
            'label': 'left'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-07-10-13-11-35_mapping_tartu_streets/camera_narrow_front/predict/crops/car/0510372.jpg',
            'label': 'left'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-04-12-13-56-08_mapping_tartu_streets/camera_fl/predict/crops/car/0018052.jpg',
            'label': 'right'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-07-11-15-51-59_mapping_tartu_streets/camera_narrow_front/predict/crops/car/074276.jpg',
            'label': 'right'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-08-16-13-56-40_mapping_tartu_streets/camera_wide_front/predict/crops/car/0660162.jpg',
            'label': 'left'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-08-16-14-33-25_mapping_tartu_streets/camera_wide_front/predict/crops/car/0879992.jpg',
            'label': 'left'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-07-11-15-51-59_mapping_tartu_streets/camera_wide_front/predict/crops/car/074317.jpg',
            'label': 'right'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-04-24-15-17-40_mapping_tartu_streets/camera_fl/predict/crops/car/002093.jpg',
            'label': 'right'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-07-16-14-21-41_mapping_tartu_streets/camera_narrow_front/predict/crops/car/055093.jpg',
            'label': 'left'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-04-09-16-18-07_mapping_tartu_streets/camera_fl/predict/crops/car/014925.jpg',
            'label': 'hazard'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-06-14-16-26-39_testing_nvidia_cam_driver_h264_encoding_for_mapping/camera_wide_left/predict/crops/car/019776.jpg',
            'label': 'none'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-07-11-15-51-59_mapping_tartu_streets/camera_narrow_front/predict/crops/car/061453.jpg',
            'label': 'none'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-09-05-14-18-54_mapping_tartu_streets_traffic_lights_ouster_lidar_2/camera_wide_front/predict/crops/car/008382.jpg',
            'label': 'none'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-09-05-14-18-54_mapping_tartu_streets_traffic_lights_ouster_lidar_2/camera_narrow_front/predict/crops/car/008724.jpg',
            'label': 'none'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-07-08-14-19-07_mapping_tartu_streets/camera_narrow_front/predict/crops/car/0906872.jpg',
            'label': 'none'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-04-18-14-25-15_mapping_tartu_streets/camera_fl/predict/crops/car/0026332.jpg',
            'label': 'none'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-09-05-14-18-54_mapping_tartu_streets_traffic_lights_ouster_lidar_2/camera_narrow_front/predict/crops/car/004698.jpg',
            'label': 'none'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-07-16-13-05-46_mapping_tartu_streets/camera_narrow_front/predict/crops/car/0075932.jpg',
            'label': 'none'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-07-11-15-39-14_mapping_tartu_streets/camera_narrow_front/predict/crops/car/052714.jpg',
            'label': 'none'
        },
        {
            'crop_path': '/gpfs/space/projects/ml2024/2024-09-05-14-18-54_mapping_tartu_streets_traffic_lights_ouster_lidar_2/camera_wide_front/predict/crops/car/005822.jpg',
            'label': 'none'
        },
    ]
    
    # Generate sequences
    sampled_df, metadata = sampler.generate_sequences(anchor_annotations)
    
    # Generate download list for HPC
    sampler.generate_download_list(sampled_df, 'sample_seq_list.json')
    
    # Generate all label formats
    sampler.create_all_labels(sampled_df, metadata, 'labels_output')
    
    # Save the full sampled dataframe for reference
    sampled_df.to_csv('sampled_frames.csv', index=False)
    print(f"\nSampled {len(sampled_df)} frames across {sampled_df['sequence_id'].nunique()} sequences")
