import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import os
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_PATH = './scripts/filtered_1010.csv'
SEQUENCE_ID_TO_TEST = '2024-03-25-15-40-16_mapping_tartu/camera_fl_2'
OUTPUT_CSV = 'cosmos_test_predictions.csv'
MODEL_ID = "nvidia/Cosmos-Reason1-7B"

# --- 1. Load Model ---
print(f"Loading {MODEL_ID}...")
try:
    # Use bfloat16 for efficiency and stability if your GPU supports it
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto" # Distributes model across available GPUs automatically
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
except Exception as e:
    print(f"FATAL ERROR: Could not load model. Check CUDA and environment setup: {e}")
    exit(1)

# --- 2. Prepare Data ---
try:
    df = pd.read_csv(CSV_PATH)
    # Filter for the specific sequence to test only a few images
    test_df = df[df['sequence_id'] == SEQUENCE_ID_TO_TEST].head(5)
    if test_df.empty:
        print(f"Error: Sequence ID '{SEQUENCE_ID_TO_TEST}' not found in CSV or is empty.")
        exit(1)
except Exception as e:
    print(f"FATAL ERROR: Could not load or filter CSV file: {e}")
    exit(1)

results = []
print(f"Processing {len(test_df)} images from sequence: {SEQUENCE_ID_TO_TEST}")

# --- 3. Inference Loop ---
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    image_path = row['crop_path'] # Full path is already in the CSV
    
    try:
        # Load Image
        image = Image.open(image_path).convert("RGB")
        
        # --- PROMPT ---
        prompt = (
            "<image>\n"
            "Analyze this image from a vehicle's perspective. "
            "Classify the turn signal into exactly one of these categories: "
            "Left, Right, Hazard, None. "
            "Answer:"
        )
        
        # Process input
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        # Generate prediction
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id # Good practice for batching later
            )
            
        # Decode and clean response
        generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
        # Simple extraction logic: find the first word matching the categories
        prediction = generated_text.split("Answer:")[-1].strip().lower()
        
        # Simple parsing to check for valid categories
        valid_labels = ['left', 'right', 'hazard', 'none']
        cleaned_prediction = next((label for label in valid_labels if label in prediction), 'unclassified')

        results.append({
            'sequence_id': row['sequence_id'],
            'crop_path': row['crop_path'],
            'true_label': row.get('true_label', 'N/A'),
            'cosmos_prediction': cleaned_prediction
        })
        
    except Exception as e:
        results.append({
            'sequence_id': row['sequence_id'],
            'crop_path': row['crop_path'],
            'cosmos_prediction': f"ERROR: {e}"
        })

# Final Save
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print("--- TEST COMPLETE ---")
print(f"Saved test results to {OUTPUT_CSV}")
