import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import os
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_PATH = '/gpfs/helios/home/dwenger/jupyter/filtered_1010.csv'
OUTPUT_CSV = 'cosmos_full_predictions.csv'
MODEL_ID = "nvidia/Cosmos-Reason1-7B"

# --- 1. Load Model (No Change) ---
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

# --- 2. Prepare Data ---
try:
    df = pd.read_csv(CSV_PATH)
    # rename test_df to processed_df and simply assign the whole dataframe.
    processed_df = df.copy()
except Exception as e:
    print(f"FATAL ERROR: Could not load CSV file: {e}")
    exit(1)

results = []
print(f"Processing {len(processed_df)} total images.")

# --- 3. Inference Loop (Using processed_df now) ---
for idx, row in tqdm(processed_df.iterrows(), total=len(processed_df)):
    image_path = row['crop_path']
    
    try:
        # Load Image
        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image from a vehicle's perspective. "
                            "Classify the turn signal into exactly one of these categories: "
                            "Left, Right, Hazard, None. "
                            "Answer:"
                        )
                    }
                ]
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate prediction
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            
        # Decode and clean response
        generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
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
print("--- FULL PROCESSING COMPLETE ---")
print(f"Saved results to {OUTPUT_CSV}")
