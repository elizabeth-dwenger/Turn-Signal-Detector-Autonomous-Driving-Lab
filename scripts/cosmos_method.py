import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import os
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_PATH = '/gpfs/helios/home/dwenger/jupyter/filtered_1010.csv'
OUTPUT_CSV = 'cosmos_temporal_predictions.csv' # Final output file
MODEL_ID = "nvidia/Cosmos-Reason1-7B"
CONTEXT_WINDOW = 8
CHECKPOINT_INTERVAL = 10000

# --- Load Model ---
print(f"Loading {MODEL_ID}...")
try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
except Exception as e:
    print(f"FATAL ERROR: Could not load model. Check CUDA and environment setup: {e}")
    exit(1)

# Function to save progress
def _save_checkpoint(current_results, filename, mode='w'):
    pd.DataFrame(current_results).to_csv(filename, index=False, mode=mode, header=(mode=='w'))
    print(f"\n--- Checkpoint saved at {len(current_results)} items to {filename} ---")


# --- Prepare Data and Checkpoint Logic ---
try:
    df = pd.read_csv(CSV_PATH)
    processed_df = df.reset_index(drop=True)
except Exception as e:
    print(f"FATAL ERROR: Could not load CSV file: {e}")
    exit(1)

results = []
start_idx = 0

# CHECKPOINT RECOVERY LOGIC
if os.path.exists(OUTPUT_CSV):
    try:
        # Read the existing results file
        results_df_old = pd.read_csv(OUTPUT_CSV)
        results = results_df_old.to_dict('records')
        
        # Determine the next image to process based on what was saved
        last_processed_path = results_df_old['crop_path'].iloc[-1]
        start_idx = processed_df[processed_df['crop_path'] == last_processed_path].index[0] + 1
        
        print(f"\n--- Resuming from checkpoint: {len(results)} items processed. Starting at index {start_idx} ---")
        
    except Exception as e:
        print(f"\n--- Warning: Could not load existing results file {OUTPUT_CSV}. Starting from scratch. ({e}) ---")
        results = []
        start_idx = 0


# Filter the dataframe to only include remaining items
processed_df_remaining = processed_df.iloc[start_idx:]

print(f"Total images to process: {len(processed_df_remaining)}")
total_processed_count = len(results)

# --- Inference Loop (Temporal Reasoning Logic) ---
# We iterate over the *remaining* dataframe using its original index
for original_idx, row in tqdm(processed_df_remaining.iterrows(), total=len(processed_df_remaining)):
    
    current_sequence_id = row['sequence_id']
    target_image_path = row['crop_path']
    
    try:
        # --- DETERMINE CONTEXT WINDOW BOUNDARIES ---
        
        # Find the start and end indices for the current sequence
        sequence_indices = processed_df[processed_df['sequence_id'] == current_sequence_id].index
        
        if sequence_indices.empty:
            raise ValueError(f"Sequence {current_sequence_id} missing or empty.")
            
        seq_start_idx = sequence_indices.min()
        seq_end_idx = sequence_indices.max()
        
        # Determine the window start and end indices
        window_start_idx = max(seq_start_idx, original_idx - CONTEXT_WINDOW)
        window_end_idx = min(seq_end_idx, original_idx + CONTEXT_WINDOW)
        
        # Get the frame paths and load images for the context window
        context_rows = processed_df.loc[window_start_idx : window_end_idx]
        context_images = []
        for context_idx, context_row in context_rows.iterrows():
            path = context_row['crop_path']
            context_images.append(Image.open(path).convert("RGB"))
            
        # CONSTRUCT MULTIMODAL PROMPT
        
        num_frames = len(context_images)
        target_frame_pos = original_idx - window_start_idx
        
        image_ref_text = []
        for i in range(num_frames):
            ref = f"Image {i+1}"
            if i == target_frame_pos:
                ref = f"Target Image {i+1}"
            image_ref_text.append(ref)
            
        image_list_str = ", ".join(image_ref_text)
        
        user_content = [{"type": "image"} for _ in range(num_frames)]
        
        prompt_text = (
            f"You are given a sequence of {num_frames} frames ({image_list_str}). "
            f"The **Target Image {target_frame_pos + 1}** is the frame requiring classification. "
            "Analyze the temporal context (frames before and after the target) to classify "
            "the car turn signal into exactly one word. "
            "You must answer with exactly ONE word from this list: left, right, hazard, none. "
            "If you are uncertain, answer 'none'. Answer:"
        )
        
        user_content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": user_content}]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        # PROCESS INPUTS
        inputs = processor(
            images=context_images,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate Prediction
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                temperature=0.0,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            
        # Decode and clean response
        generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
        prediction = generated_text.split("Answer:")[-1].strip().lower()
        
        valid_labels = ['left', 'right', 'hazard', 'none']
        cleaned_prediction = prediction.strip().lower()

        if cleaned_prediction not in valid_labels:
             cleaned_prediction = "unclassified"

        # Append result to list
        results.append({
            'sequence_id': row['sequence_id'],
            'frame_id': row['frame_id'],
            'crop_path': target_image_path,
            'context_window_size': num_frames,
            'true_label': row.get('true_label', 'N/A'),
            'cosmos_prediction': cleaned_prediction
        })
        
    except Exception as e:
        # Append error result
        results.append({
            'sequence_id': current_sequence_id,
            'crop_path': target_image_path,
            'cosmos_prediction': f"ERROR: {e}"
        })
    
    # --- CHECKPOINTING ---
    total_processed_count += 1
    if total_processed_count % CHECKPOINT_INTERVAL == 0:
        _save_checkpoint(results, OUTPUT_CSV)


# Final Save after the loop finishes
_save_checkpoint(results, OUTPUT_CSV)
print("--- FULL TEMPORAL PROCESSING COMPLETE ---")
print(f"Saved final results to {OUTPUT_CSV}")
