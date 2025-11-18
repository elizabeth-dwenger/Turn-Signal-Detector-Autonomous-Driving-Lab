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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_image_grid(image_paths: List[str], max_cols: int = 5) -> Image.Image:
    """
    Concatenate multiple images into a grid.
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


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_sequence_with_result(
    image_paths: List[str],
    result: Dict,
    true_label: str = None,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Visualize a sequence of images with API prediction results.
    """
    n_images = len(image_paths)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    # Plot each image
    for idx, (ax, img_path) in enumerate(zip(axes_flat[:n_images], image_paths)):
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            
            # Highlight frames with signal (if provided)
            if 'frames_with_signal' in result and idx in result['frames_with_signal']:
                # Add green border for frames with visible signal
                rect = Rectangle((0, 0), img.width-1, img.height-1, 
                               linewidth=4, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.set_title(f"Frame {idx + 1}\n✓ Signal", 
                           color='green', fontweight='bold', fontsize=10)
            else:
                ax.set_title(f"Frame {idx + 1}", fontsize=10)
            
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{Path(img_path).name}", 
                   ha='center', va='center')
            ax.axis('off')
    
    # Hide extra subplots
    for ax in axes_flat[n_images:]:
        ax.axis('off')
    
    # Add result information as text
    result_text = f"PREDICTION RESULTS\n{'='*50}\n\n"
    
    if true_label:
        match = "✓ CORRECT" if result.get('label') == true_label else "✗ INCORRECT"
        color = 'green' if result.get('label') == true_label else 'red'
        result_text += f"True Label: {true_label.upper()}\n"
        result_text += f"Predicted: {result.get('label', 'N/A').upper()} ({match})\n\n"
    else:
        result_text += f"Predicted: {result.get('label', 'N/A').upper()}\n\n"
    
    result_text += f"Confidence: {result.get('confidence', 0):.2%}\n\n"
    result_text += f"Method: {result.get('method', 'N/A')}\n\n"
    
    if 'reasoning' in result:
        wrapped_reasoning = textwrap.fill(result['reasoning'], width=50)
        result_text += f"Reasoning:\n{wrapped_reasoning}\n\n"
    
    if 'frames_with_signal' in result:
        result_text += f"Signal visible in frames: {result['frames_with_signal']}\n\n"
    
    
    # Add text box with results
    fig.text(0.02, 0.98, result_text, 
            verticalalignment='top',
            fontsize=10,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def visualize_comparison(
    image_paths: List[str],
    individual_result: Dict,
    grid_result: Dict,
    true_label: str = None,
    figsize: Tuple[int, int] = (18, 12)
):
    """
    Visualize sequence with both individual and grid method results side by side.
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[0.15, 1, 0.5], hspace=0.3, wspace=0.2)
    
    # Title
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title_text = f"SEQUENCE COMPARISON"
    if true_label:
        title_text += f" | True Label: {true_label.upper()}"
    title_ax.text(0.5, 0.5, title_text, ha='center', va='center', 
                 fontsize=16, fontweight='bold')
    
    # Individual method images
    individual_ax = fig.add_subplot(gs[1, 0])
    individual_ax.axis('off')
    individual_ax.set_title("INDIVIDUAL FRAMES METHOD", fontweight='bold', fontsize=12)
    
    # Create mini grid for individual
    n_images = len(image_paths)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    mini_fig_individual = create_image_grid(image_paths, max_cols=n_cols)
    individual_ax.imshow(mini_fig_individual)
    
    # Grid method image
    grid_ax = fig.add_subplot(gs[1, 1])
    grid_ax.axis('off')
    grid_ax.set_title("GRID METHOD", fontweight='bold', fontsize=12)
    
    grid_image = create_image_grid(image_paths, max_cols=min(5, n_images))
    grid_ax.imshow(grid_image)
    
    # Individual results text
    individual_text_ax = fig.add_subplot(gs[2, 0])
    individual_text_ax.axis('off')
    
    ind_text = format_result_text(individual_result, true_label, "Individual")
    individual_text_ax.text(0, 1, ind_text, 
                           verticalalignment='top',
                           fontsize=10,
                           family='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Grid results text
    grid_text_ax = fig.add_subplot(gs[2, 1])
    grid_text_ax.axis('off')
    
    grid_text = format_result_text(grid_result, true_label, "Grid")
    grid_text_ax.text(0, 1, grid_text,
                     verticalalignment='top',
                     fontsize=10,
                     family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
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
    
    text += f"Confidence: {result.get('confidence', 0):.2%}\n\n"
    
    if 'reasoning' in result:
        wrapped = textwrap.fill(result['reasoning'], width=35)
        text += f"Reasoning:\n{wrapped}\n\n"
    
    if 'frames_with_signal' in result:
        text += f"Signal frames: {result['frames_with_signal']}\n\n"
    
    if 'error' in result:
        text += f"\nERROR: {result['error']}\n"
    
    return text


def visualize_all_sequences_summary(results_df: pd.DataFrame, output_path: str = None):
    """
    Create a summary visualization showing all sequences and their predictions.
    """
    n_sequences = len(results_df)
    
    fig, axes = plt.subplots(n_sequences, 3, figsize=(15, n_sequences * 2))
    if n_sequences == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        # Column 1: First frame thumbnail
        try:
            first_frame = Image.open(row['frame_paths'][0])
            axes[idx, 0].imshow(first_frame)
            axes[idx, 0].set_title(f"{row['sequence_id']}\nTrue: {row['true_label']}", 
                                  fontsize=9)
        except:
            axes[idx, 0].text(0.5, 0.5, "No image", ha='center', va='center')
        axes[idx, 0].axis('off')
        
        # Column 2: Individual method result
        if 'individual' in row and isinstance(row['individual'], dict):
            ind_pred = row['individual'].get('label', 'error')
            ind_conf = row['individual'].get('confidence', 0)
            color = 'green' if ind_pred == row['true_label'] else 'red'
            axes[idx, 1].text(0.5, 0.5, 
                            f"Individual:\n{ind_pred.upper()}\n{ind_conf:.0%}",
                            ha='center', va='center', fontsize=11,
                            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        else:
            axes[idx, 1].text(0.5, 0.5, "No result", ha='center', va='center')
        axes[idx, 1].axis('off')
        
        # Column 3: Grid method result
        if 'grid' in row and isinstance(row['grid'], dict):
            grid_pred = row['grid'].get('label', 'error')
            grid_conf = row['grid'].get('confidence', 0)
            color = 'green' if grid_pred == row['true_label'] else 'red'
            axes[idx, 2].text(0.5, 0.5,
                            f"Grid:\n{grid_pred.upper()}\n{grid_conf:.0%}",
                            ha='center', va='center', fontsize=11,
                            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        else:
            axes[idx, 2].text(0.5, 0.5, "No result", ha='center', va='center')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Summary saved to {output_path}")
    
    return fig


# ============================================================================
# OPENAI API LABELING FUNCTIONS
# ============================================================================

def label_sequence_individual_frames(
    image_paths: List[str],
    model: str = "gpt-4o-mini",
    preprocess: str = "none"
) -> Dict:
    """
    Send individual frames to OpenAI API for labeling using JSON mode.
    """

    # Optional preprocessing imports
    if preprocess != "none":
        from turn_signal_preprocessing import (
            enhance_image,
            isolate_yellow_channel,
            analyze_sequence_for_turn_signals
        )

    # Base prompt
    base_prompt = """
Analyze this sequence of car images taken from a vehicle's camera. 
The images are in temporal order and show a tracked car.

Your task: Determine if the car's turn signals are active and which type:
- "left": Left turn signal is blinking
- "right": Right turn signal is blinking  
- "hazard": Both turn signals are blinking (hazard lights)
- "none": No turn signals are active

Important notes:
- Turn signals BLINK on and off, so you may see the light in some frames but not others.
- Look for amber/orange lights on the sides or rear of the vehicle.
- The pattern of on/off across frames indicates blinking.
- Hazard lights mean BOTH left AND right blink together.
"""

    # Add preprocessing description
    if preprocess == "enhanced":
        base_prompt += """

PREPROCESSING NOTE:
Images were enhanced (gamma + CLAHE). Active signals will appear VERY bright."""
    elif preprocess == "yellow":
        base_prompt += """

PREPROCESSING NOTE:
Images display ONLY yellow/amber channels. White/bright clusters indicate turn signals. This means if there are more white on the left on some frames, likely the turn signal is left. If there is more right on some frames, then it is likely a right turn signal."""
    elif preprocess == "analysis":
        analysis = analyze_sequence_for_turn_signals(image_paths, fps=5.0, visualize=False)
        base_prompt += f"""

PREPROCESSING ANALYSIS RESULTS:
- Periodic signal detected: {analysis['periodic_analysis']['is_periodic']}
- Peak frequency: {analysis['periodic_analysis']['peak_frequency']:.2f} Hz
- Predicted signal: {analysis['predicted_signal']}

Use this analysis to guide your visual inspection.
"""

    # Required JSON response format
    base_prompt += """
Respond ONLY with valid JSON:
{
  "label": "left" | "right" | "hazard" | "none",
  "confidence": float,
  "reasoning": "Brief explanation",
  "frames_with_signal": [frame indices]
}
"""

    content = [{"type": "text", "text": base_prompt}]

    # for idx, path in enumerate(image_paths):
    #     # Preprocessing
    #     if preprocess in ("enhanced", "yellow"):
    #         img = cv2.imread(path)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #         if preprocess == "enhanced":
    #             img = enhance_image(img, gamma=2.0)
    #         elif preprocess == "yellow":
    #             yellow_mask = isolate_yellow_channel(img)
    #             heat = cv2.applyColorMap(yellow_mask, cv2.COLORMAP_HOT)
    #             img = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    #         pil = Image.fromarray(img)
    #         base64_img = encode_pil_image_to_base64(pil)

    #     else:
    #         base64_img = encode_image_to_base64(path)

    #     # Add image
    #     content.append({
    #         "type": "image_url",
    #         "image_url": {
    #             "url": f"data:image/jpeg;base64,{base64_img}"
    #         }
    #     })

    #     # Frame index label
    #     content.append({"type": "text", "text": f"Frame {idx+1}/{len(image_paths)}"})


    # --- API CALL (JSON MODE) ---
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
            temperature=0.1
        )

        # JSON mode guarantees valid JSON
        text = response.choices[0].message.content
        obj = json.loads(text)

        return {
            **obj,
            "method": "individual_frames",
            "preprocess": preprocess,
            "num_frames": len(image_paths),
            "model": model,
        }

    except Exception as e:
        return {
            "label": "error",
            "error": str(e),
            "method": "individual_frames",
            "preprocess": preprocess
        }


def label_sequence_grid(
    image_paths: List[str],
    model: str = "gpt-4o-mini",
    max_cols: int = 5
) -> Dict:

    grid_img = create_image_grid(image_paths, max_cols=max_cols)
    base64_grid = encode_pil_image_to_base64(grid_img)

    prompt = f"""Analyze this grid of {len(image_paths)} sequential car images (left-to-right, top-to-bottom).
These frames show a tracked car over time from a vehicle's camera.

Your task: Determine if the car's turn signals are active:
- "left": Left turn signal blinking
- "right": Right turn signal blinking
- "hazard": Both signals blinking together
- "none": No turn signals

Key considerations:
- Turn signals BLINK, so some frames will show the light while others will not.
- Look for amber/orange regions or bright clusters indicating a turn signal.
- Hazard lights = both sides blink in the same pattern.

Respond ONLY with valid JSON:
{{
  "label": "left" | "right" | "hazard" | "none",
  "confidence": float,
  "reasoning": "Brief explanation",
  "frames_with_signal": [list of frame indices]
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=500
        )

        # JSON mode guarantees valid JSON
        response_text = response.choices[0].message.content
        result = json.loads(response_text)

        # Add metadata
        result["method"] = "grid"
        result["num_frames"] = len(image_paths)
        result["model"] = model

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
    image_base_path: str = "../seq_img",  # Base path where images are stored
    preprocess: str = "none"  # Preprocessing: "none", "enhanced", "yellow", "analysis"
) -> pd.DataFrame:
    """
    Process all sequences from your label CSV using OpenAI API.
    """
    
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
        
        # Process with requested method(s)
        if method in ["individual", "both"]:
            print(f"  Running individual frames method (preprocess={preprocess})...")
            individual_result = label_sequence_individual_frames(
                image_paths, 
                model=model,
                preprocess=preprocess
            )
            result_entry['individual'] = individual_result
        
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


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_results(results_df: pd.DataFrame) -> Dict:
    """
    Compare OpenAI predictions against true labels.
    """
    
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