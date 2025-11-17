import os
import json
import base64
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Tuple
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def encode_image_to_base64(image_path: str) -> str:
    # Encode image to base64 string for API
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_image_grid(image_paths: List[str], max_cols: int = 5) -> Image.Image:
    images = [Image.open(path).convert('RGB') for path in image_paths]
    
    # Get dimensions (assume all images same size)
    img_width, img_height = images[0].size
    
    # Calculate grid dimensions
    n_images = len(images)
    n_cols = min(max_cols, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create blank canvas
    grid_width = n_cols * img_width
    grid_height = n_rows * img_height
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))
    
    return grid


def encode_pil_image_to_base64(image: Image.Image) -> str:
    """Encode PIL image to base64."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ============================================================================
# OPENAI API LABELING FUNCTIONS
# ============================================================================

def label_sequence_individual_frames(
    image_paths: List[str],
    model: str = "gpt-4o-mini"
) -> Dict:

    # JSON schema to force valid output
    json_schema = {
        "name": "turn_signal_schema",
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "reasoning": {"type": "string"},
                "frames_with_signal": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "required": ["label", "reasoning", "frames_with_signal"]
        }
    }

    content = [
        {
            "type": "text",
            "text": f"""Analyze this sequence of car images taken from a vehicle's camera. 
The images are in temporal order and show a tracked car.

Your task: Determine if the car's turn signals are active by looking at the differences between frames:
- "left": Amber colored turn light on LEFT side of car is blinking in some frames
- "right": Amber colored turn light on RIGHT side of car is blinking in some frames
- "hazard": Both turn signals are blinking (hazard lights) in some frames. If amber colored lights are on in all frames, this is "none", not "hazard"
- "none": No turn signals are active in any frames

Important notes:
- Turn signals BLINK on and off, so you SHOULD see the light in some frames but not others
- Look for amber/orange lights at the rear of the vehicle
- The pattern of on/off across frames indicates blinking
- Hazard lights mean BOTH left AND right are blinking together

Respond ONLY with valid JSON in this exact format:
{json_schema}"""
        }
    ]

    for idx, image_path in enumerate(image_paths):
        base64_image = encode_image_to_base64(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        })
        content.append({
            "type": "text",
            "text": f"Frame {idx}"
        })

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=500,
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            }
        )

        response_text = response.choices[0].message.content

        result = json.loads(response_text)

        # Metadata
        result["method"] = "individual_frames"
        result["num_frames"] = len(image_paths)
        result["model"] = model
        result["tokens_used"] = {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }

        return result

    except Exception as e:
        return {
            "label": "error",
            "error": str(e),
            "method": "individual_frames"
        }

def label_sequence_grid(
    image_paths: List[str],
    model: str = "gpt-4o-mini",
    max_cols: int = 5
) -> Dict:

    # Required JSON schema format
    json_schema = {
        "name": "turn_signal_schema_grid",
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "reasoning": {"type": "string"},
                "frames_with_signal": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "required": ["label", "reasoning", "frames_with_signal"]
        }
    }

    # Create the image grid
    grid_image = create_image_grid(image_paths, max_cols=max_cols)
    base64_grid = encode_pil_image_to_base64(grid_image)
    
    prompt = f"""Analyze this grid of {len(image_paths)} sequential car images (left-to-right, top-to-bottom).
The images show a tracked car over time from a vehicle's camera.

Your task: Determine if the car's turn signals are active by looking at the differences between frames:
- "left": Amber colored turn light on LEFT side of car is blinking in some frames
- "right": Amber colored turn light on RIGHT side of car is blinking in some frames
- "hazard": Both turn signals are blinking (hazard lights) in some frames. If amber colored lights are on in all frames, this is "none", not "hazard"
- "none": No turn signals are active in any frames

Important notes:
- Turn signals BLINK on and off, so you SHOULD see the light in some frames but not others
- Look for amber/orange lights at the rear of the vehicle
- The pattern of on/off across frames indicates blinking
- Hazard lights mean BOTH left AND right are blinking together

Respond ONLY with valid JSON in this exact format:
{json_schema}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_grid}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            }
        )

        # Parse the model output
        response_text = response.choices[0].message.content
        result = json.loads(response_text)

        # Add metadata
        result["method"] = "grid"
        result["num_frames"] = len(image_paths)
        result["model"] = model
        result["tokens_used"] = {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }

        return result

    except Exception as e:
        return {
            "label": "error",
            "error": str(e),
            "method": "grid"
        }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_all_sequences(
    label_csv: str,
    method: str = "individual",  # "individual" or "grid" or "both"
    model: str = "gpt-4o-mini",
    output_path: str = "openai_labels.json",
    delay: float = 1.0,  # Delay between API calls to avoid rate limits
    image_base_path: str = "../seq_img"  # Base path where images are stored
) -> pd.DataFrame:
    
    # Load labels
    df = pd.read_csv(label_csv)
    
    # Prepend base path to crop_path if needed
    if image_base_path:
        df['crop_path'] = df['crop_path'].apply(
            lambda x: str(image_base_path + x) if not Path(x).exists() else x
        )
    
    # Group by sequence
    results = []
    
    sequences = df.groupby('sequence_id')
    total_sequences = len(sequences)
    
    print(f"Processing {total_sequences} sequences using method: {method}")
    print(f"Model: {model}")
    print(f"Image base path: {image_base_path}")
    print("-" * 60)
    
    for idx, (seq_id, group) in enumerate(sequences):
        print(f"\n[{idx+1}/{total_sequences}] Processing {seq_id}...")
        
        # Get frames in order
        group = group.sort_values('frame_idx')
        image_paths_all = group['crop_path'].tolist()
        image_paths = image_paths_all[:4]   # use only the first 4 frames
        true_label = group['label'].iloc[0]
        
        # Verify first image exists
        if not Path(image_paths[0]).exists():
            print(f"  WARNING: Image not found: {image_paths[0]}")
            print(f"  Skipping sequence...")
            continue
        
        print(f"  True label: {true_label}")
        print(f"  Frames: {len(image_paths)}")
        print(f"  First frame: {Path(image_paths[0]).name}")
        
        result_entry = {
            'sequence_id': seq_id,
            'true_label': true_label,
            'num_frames': len(image_paths),
            'frame_paths': image_paths
        }
        
        # Process with requested method(s)
        if method in ["individual", "both"]:
            print(f"  Running individual frames method...")
            individual_result = label_sequence_individual_frames(image_paths, model=model)
            result_entry['individual'] = individual_result
            
            if 'tokens_used' in individual_result:
                tokens = individual_result['tokens_used']['total']
                print(f"    Predicted: {individual_result.get('label', 'error')}")
        
        if method in ["grid", "both"]:
            print(f"  Running grid method...")
            grid_result = label_sequence_grid(image_paths, model=model)
            result_entry['grid'] = grid_result
            
            if 'tokens_used' in grid_result:
                tokens = grid_result['tokens_used']['total']
                print(f"    Predicted: {grid_result.get('label', 'error')}")
        
        results.append(result_entry)
        
        # Save intermediate results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Rate limiting
        if idx < total_sequences - 1:
            time.sleep(delay)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_path}")
    
    return pd.DataFrame(results)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_results(results_df: pd.DataFrame) -> Dict:    
    metrics = {}
    
    # Evaluate individual frames method if present
    if 'individual' in results_df.columns:
        individual_preds = results_df['individual'].apply(
            lambda x: x.get('label', 'error') if isinstance(x, dict) else 'error'
        )
        individual_correct = (individual_preds == results_df['true_label']).sum()
        individual_accuracy = individual_correct / len(results_df)
        
        metrics['individual'] = {
            'accuracy': individual_accuracy,
            'correct': int(individual_correct),
            'total': len(results_df),
            'confusion_matrix': pd.crosstab(
                results_df['true_label'], 
                individual_preds,
                rownames=['True'],
                colnames=['Predicted']
            ).to_dict()
        }
        
        print("Individual Frames Method:")
        print(f"  Accuracy: {individual_accuracy:.2%} ({individual_correct}/{len(results_df)})")
        print(f"  Confusion Matrix:")
        print(pd.crosstab(results_df['true_label'], individual_preds))
        print()
    
    # Evaluate grid method if present
    if 'grid' in results_df.columns:
        grid_preds = results_df['grid'].apply(
            lambda x: x.get('label', 'error') if isinstance(x, dict) else 'error'
        )
        grid_correct = (grid_preds == results_df['true_label']).sum()
        grid_accuracy = grid_correct / len(results_df)
        
        metrics['grid'] = {
            'accuracy': grid_accuracy,
            'correct': int(grid_correct),
            'total': len(results_df),
            'confusion_matrix': pd.crosstab(
                results_df['true_label'],
                grid_preds,
                rownames=['True'],
                colnames=['Predicted']
            ).to_dict()
        }
        
        print("Grid Method:")
        print(f"  Accuracy: {grid_accuracy:.2%} ({grid_correct}/{len(results_df)})")
        print(f"  Confusion Matrix:")
        print(pd.crosstab(results_df['true_label'], grid_preds))
        print()
    
    return metrics