import os
import json
import base64
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Tuple
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_image_grid(image_paths: List[str], max_cols: int = 5) -> Image.Image:
    """
    Concatenate multiple images into a grid.
    
    Args:
        image_paths: List of paths to images
        max_cols: Maximum number of columns in grid
        
    Returns:
        PIL Image of the grid
    """
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


def visualize_sequence_with_result(
    image_paths,
    result,
    true_label=None,
    figsize=(16, 10)
):
    n_images = len(image_paths)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(n_rows, n_cols)
    axes_flat = axes.flatten()

    for idx, (ax, img_path) in enumerate(zip(axes_flat, image_paths)):
        img = Image.open(img_path)
        ax.imshow(img)

        if 'frames_with_signal' in result and idx in result['frames_with_signal']:
            rect = Rectangle((0, 0), img.width-1, img.height-1, 
                             linewidth=4, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(f"Frame {idx+1}\nSignal", color='green', fontsize=10)
        else:
            ax.set_title(f"Frame {idx+1}", fontsize=10)

        ax.axis('off')

    # Hide unused axes
    for ax in axes_flat[n_images:]:
        ax.axis('off')

    result_text = f"Predicted: {result.get('label','N/A')}\n"
    if true_label:
        result_text += f"True Label: {true_label}\n"
    if 'reasoning' in result:
        result_text += f"\nReasoning:\n{result['reasoning']}"

    fig.text(0.02, 0.98, result_text, va='top', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def visualize_comparison(
    image_paths,
    individual_result,
    grid_result,
    true_label=None,
    figsize=(18, 12)
):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.15, 1, 0.5], hspace=0.3, wspace=0.2)

    # Title
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title = "SEQUENCE COMPARISON"
    if true_label:
        title += f" | True: {true_label}"
    title_ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=16)

    # Image grids
    ind_ax = fig.add_subplot(gs[1, 0])
    ind_ax.axis('off')
    ind_ax.imshow(create_image_grid(image_paths))

    grid_ax = fig.add_subplot(gs[1, 1])
    grid_ax.axis('off')
    grid_ax.imshow(create_image_grid(image_paths))

    # Text
    ind_text_ax = fig.add_subplot(gs[2, 0]); ind_text_ax.axis('off')
    grid_text_ax = fig.add_subplot(gs[2, 1]); grid_text_ax.axis('off')

    ind_text = f"Individual Prediction: {individual_result.get('label')}"
    grid_text = f"Grid Prediction: {grid_result.get('label')}"

    ind_text_ax.text(0, 1, ind_text, va='top', fontsize=10)
    grid_text_ax.text(0, 1, grid_text, va='top', fontsize=10)

    return fig


def format_result_text(result: Dict, true_label: str = None, method_name: str = "") -> str:
    """Format result dictionary as readable text."""
    text = f"{method_name.upper()} RESULTS\n{'='*40}\n\n"
    
    predicted = result.get('label', 'error').upper()
    
    if true_label:
        match = "✓ CORRECT" if result.get('label') == true_label else "✗ INCORRECT"
        text += f"True: {true_label.upper()}\n"
        text += f"Predicted: {predicted} ({match})\n\n"
    else:
        text += f"Predicted: {predicted}\n\n"
        
    if 'reasoning' in result:
        wrapped = textwrap.fill(result['reasoning'], width=35)
        text += f"Reasoning:\n{wrapped}\n\n"
    
    if 'frames_with_signal' in result:
        text += f"Signal frames: {result['frames_with_signal']}\n\n"
    
    
    if 'error' in result:
        text += f"\nERROR: {result['error']}\n"
    
    return text


def visualize_all_sequences_summary(results_df, output_path=None):
    n = len(results_df)
    fig, axes = plt.subplots(n, 3, figsize=(15, n * 2))
    axes = np.array(axes).reshape(n, 3)

    for i, (_, row) in enumerate(results_df.iterrows()):
        # Column 1
        img = Image.open(row['frame_paths'][0])
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(row['sequence_id'])
        axes[i, 0].axis('off')

        # Column 2
        axes[i, 1].text(0.5, 0.5, str(row['individual'].get('label')), 
                        ha='center', va='center')
        axes[i, 1].axis('off')

        # Column 3
        axes[i, 2].text(0.5, 0.5, str(row['grid'].get('label')), 
                        ha='center', va='center')
        axes[i, 2].axis('off')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


# ============================================================================
# OPENAI API LABELING FUNCTIONS
# ============================================================================

def label_sequence_with_summary_image(
    image_paths: List[str],
    model: str = "gpt-4o",  # multimodal GPT-4o
    include_analysis: bool = True
) -> Dict:
    """
    Send a SINGLE comprehensive summary image instead of multiple frames.
    This gives the API all information at once and reduces token cost.
    """
    from advanced_preprocessing import create_comprehensive_summary_image
    from turn_signal_preprocessing import analyze_sequence_for_turn_signals

    # Run computational analysis
    analysis = None
    if include_analysis:
        analysis = analyze_sequence_for_turn_signals(
            image_paths, fps=5.0, visualize=False
        )

    # Create comprehensive summary image
    summary_img = create_comprehensive_summary_image(image_paths, analysis)

    # Convert to PIL and encode as base64
    pil_img = Image.fromarray(summary_img)
    base64_image = encode_pil_image_to_base64(pil_img)

    # Create detailed prompt
    prompt = """Analyze this comprehensive turn signal detection summary image.

The image shows:
- TOP: Grid of sequential frames from a car's rear view (numbered)
- BOTTOM LEFT: "Changes Over Time" - heatmap showing where pixels change (bright = more change)
- BOTTOM RIGHT: Graph showing yellow/amber light intensity across frames

Your task: Determine the turn signal status:
- "left": Left turn signal is blinking
- "right": Right turn signal is blinking  
- "hazard": Both signals blinking (hazard lights)
- "none": No turn signals active

Key indicators:
- Turn signals BLINK at ~1-2 Hz (60-120 times/min)
- Look for periodic spikes in the intensity graph
- Changes heatmap shows WHERE blinking occurs (left, right, or both sides)
- Turn signals are amber/orange colored
"""

    if include_analysis and analysis:
        prompt += f"""

COMPUTATIONAL ANALYSIS RESULTS (use as guidance):
- Periodic signal detected: {analysis['periodic_analysis']['is_periodic']}
- Blink frequency: {analysis['periodic_analysis']['blinks_per_minute']:.0f} bpm
- Predicted signal: {analysis['predicted_signal']}

Use your visual analysis to VERIFY or CORRECT these computational predictions.
"""

    prompt += """

Respond ONLY with valid JSON:
{
    "label": "left" | "right" | "hazard" | "none",
    "reasoning": "What you observed in the frames, heatmap, and graph",
    "agrees_with_analysis": true/false (if analysis provided)
}"""

    try:
        # New multimodal usage: input can include text + image
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_data": base64_image,
                            "image_format": "jpeg"
                        }
                    ],
                }
            ],
            max_output_tokens=500,
            temperature=0.1
        )

        response_text = response.choices[0].message.content
        result = json.loads(response_text)

        # Add metadata
        result.update({
            "method": "summary_image",
            "num_frames": len(image_paths),
            "model": model,
            "included_analysis": include_analysis,
            "tokens_used": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
        })

        if include_analysis and analysis:
            result["computational_prediction"] = analysis['predicted_signal']

        return result

    except Exception as e:
        return {
            "label": "error",
            "error": str(e),
            "method": "summary_image"
        }



def label_sequence_grid(
    image_paths: List[str],
    model: str = "gpt-4o-mini",
    max_cols: int = 5
) -> Dict:
    """
    Concatenate frames into a grid and send to OpenAI API.
    This is cheaper (1 image vs N images) but may be less accurate.
    
    Args:
        image_paths: List of image paths in temporal order
        model: OpenAI model to use
        max_cols: Maximum columns in grid
        
    Returns:
        Dictionary with label and reasoning
    """
    
    # Create grid
    grid_image = create_image_grid(image_paths, max_cols=max_cols)
    base64_grid = encode_pil_image_to_base64(grid_image)
    
    prompt = f"""Analyze this grid of {len(image_paths)} sequential car images (left-to-right, top-to-bottom).
The images show a tracked car over time from a vehicle's camera.

Your task: Determine if the car's turn signals are active:
- "left": Left turn signal is blinking
- "right": Right turn signal is blinking
- "hazard": Both turn signals are blinking (hazard lights)
- "none": No turn signals are active

Important notes:
- Turn signals BLINK on and off across the sequence
- Look for amber/orange lights on sides or rear
- The on/off pattern across frames indicates blinking
- Hazard lights mean BOTH sides blink together

Respond ONLY with valid JSON:
{{
    "label": "left" | "right" | "hazard" | "none",
    "reasoning": "Brief explanation",
    "frames_with_signal": [frame numbers where signal was visible]
}}"""
    
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
            temperature=0.1
        )
        
        response_text = response.content[0].text
        result = json.loads(response_text)
        
        result['method'] = 'grid'
        result['num_frames'] = len(image_paths)
        result['model'] = model
        result['tokens_used'] = {
            'prompt': response.usage.prompt_tokens,
            'completion': response.usage.completion_tokens,
            'total': response.usage.total_tokens
        }
        
        return result
        
    except Exception as e:
        return {
            'label': 'error',
            'error': str(e),
            'method': 'grid'
        }


def process_all_sequences(
    label_csv: str,
    method: str = "individual",  # "individual" or "grid" or "both"
    model: str = "gpt-4o-mini",
    output_path: str = "openai_labels.json",
    delay: float = 1.0,  # Delay between API calls to avoid rate limits
    image_base_path: str = "../seq_img",  # Base path where images are stored
    preprocess: str = "none"  # Preprocessing: "none", "enhanced", "yellow", "analysis"
) -> pd.DataFrame:
    
    # Load labels
    df = pd.read_csv(label_csv)
    
    # Prepend base path to crop_path if needed
    if image_base_path:
        df['crop_path'] = df['crop_path'].apply(
            lambda x: str(Path(image_base_path) / x) if not Path(x).exists() else x
        )
    
    # Group by sequence
    results = []
    
    sequences = df.groupby('sequence_id')
    total_sequences = len(sequences)
    
    print(f"Processing {total_sequences} sequences using method: {method}")
    print(f"Model: {model}")
    print(f"Image base path: {image_base_path}")
    print(f"Preprocessing: {preprocess}")
    print("-" * 60)
    
    for idx, (seq_id, group) in enumerate(sequences):
        print(f"\n[{idx+1}/{total_sequences}] Processing {seq_id}...")
        
        # Get frames in order
        group = group.sort_values('frame_idx')
        image_paths = group['crop_path'].tolist()
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
            
        
        if method in ["grid", "both"]:
            print(f"  Running grid method...")
            grid_result = label_sequence_grid(image_paths, model=model)
            result_entry['grid'] = grid_result
            
        
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Process all sequences with both methods
    results_df = process_all_sequences(
        label_csv='labels_output/labels_lstm.csv',
        method='both',  # Try both individual and grid
        model='gpt-4o-mini',  # Use gpt-4o for better accuracy
        output_path='openai_labels_results.json',
        delay=1.0,
        image_base_path='../seq_img'  # Adjust this to your image location
    )
    
    # Evaluate
    metrics = evaluate_results(results_df)
    
    # Save metrics
    with open('openai_evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nEvaluation complete!")
    print(f"Results: openai_labels_results.json")
    print(f"Metrics: openai_evaluation_metrics.json")
