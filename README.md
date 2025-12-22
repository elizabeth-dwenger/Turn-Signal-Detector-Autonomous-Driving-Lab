# 🚗 Turn Signal Dataset Project

This repository contains code and documentation for creating a dataset of vehicle light states (left/right turn signals, hazard, tail light, tail light off, and no visible tail lights).  

---

## Project Setup

### Create and Activate a Python Environment

I use `pyenv` to manage Python versions and virtual environments.

```
# Install and select Python version (if not already installed)
pyenv install 3.12.6
pyenv virtualenv 3.12.6 adl
pyenv local adl
```

### Install dependencies

```
pip install -r requirements.txt
```

### Directory Structure

```
ADL/
├── jupyter/                 # Jupyter notebooks for exploration
├── sampled_images/          # Locally sampled images (ignored by Git)
├── front_back_images/       # Sampled images for front/back classification
├── .env                     # API keys (not tracked by Git)
├── .gitignore
├── requirements.txt
└── README.md
```

### API Key Setup

1. Create a `.env` file in the project root
2. Add: `OPENAI_API_KEY=sk-your-key-here`
3. Ensure `.env` is listed in `.gitignore`

---

## JSON Annotations Format

```json
[
  {
    "image": "../data_sample/car_001.jpg",
    "turn_signal": "left",
    "tail_light": "on"
  },
  {
    "image": "../data_sample/car_002.jpg",
    "turn_signal": "none",
    "tail_light": "not_visible"
  },
  {
    "image": "../data_sample/car_003.jpg",
    "turn_signal": "both",
    "tail_light": "off"
  }
]
```

---

### Schema Definition

**Turn Signal (`turn_signal`)**

* `left`
* `right`
* `none`
* `both` (hazard)
* `unclear`

**Tail Light (`tail_light`)**

* `on`
* `off`
* `not_visible`
* `unclear`

---

## Data Access & Sampling

### Counting total number of cropped car images

```
for dir in 2024-*_mapping_tartu*; do
  if [ -d "$dir" ]; then
    session_total=0
    while IFS= read -r car_dir; do
      count=$(find "$car_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l)
      session_total=$((session_total + count))
    done < <(find "$dir" -type d -path "*/predict/crops/car")
    
    printf '%s: %d\n' "$dir" "$session_total"
    total=$((total + session_total))
  fi
done

printf 'TOTAL: %d\n' "$total"
```

**Total car crops as of September 2025:** `8,790,531`

---

## Automated Filtering & Sampling

### Filter by size and orientation

Keep images that are **landscape (width > height)** and **≥ 50x50 pixels**:

```
ssh user@cluster.example.edu "
  find /data/projects/ml2024/2024-* -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) \
  | grep 'predict/crops/car' \
  | while read -r img; do
      dims=\$(identify -format '%w %h' \"\$img\" 2>/dev/null) || continue
      set -- \$dims
      w=\$1
      h=\$2
      if [ \"\$w\" -gt 50 ] && [ \"\$h\" -gt 50 ] && [ \"\$w\" -gt \"\$h\" ]; then
        echo \"\$img\"
      fi
    done > /home/user/filtered_landscape_images.txt
"
```

**Filtered landscape images:** `4,731,786`

---

### Sample subset

```
shuf -n 10000 filtered_landscape_images.txt > sample_10k.txt
scp user@cluster.example.edu:/home/user/sample_10k.txt ./
rsync -av --files-from=sample_10k.txt user@cluster.example.edu:/ front_back_images/
```

---

## Front/Back Classification Model

### Training (ResNet-18)

```
caffeinate python scripts/train_binary.py \
  --train-file train_front_back.txt \
  --val-file val_front_back.txt \
  --img-root front_back_images \
  --epochs 8 \
  --batch-size 32 \
  --lr 1e-4 \
  --save-dir runs/resnet18_expt
```

**Results:**

```
Epoch 8/8 | time 83.1s | train_loss 0.0108 | val_loss 0.1592 | val_acc 0.9520
precision 0.9300 | recall 0.9151 | f1 0.9225
Best val_acc: 0.9575
```

---

## Inference Pipeline

```
python /home/user/scripts/infer_all_front_back.py \
  --checkpoint /home/user/runs/resnet18_expt/best_model.pth \
  --image-list /home/user/tmp/test_5_images.txt \
  --img-root /data/projects/ml2024/ \
  --output /home/user/tmp/test_infer_5_out.txt \
  --device cuda
```

---

## Running on GPU

### `run_infer_all_front_back.sbatch`

```
#!/bin/bash
#SBATCH --job-name=back_of_car_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1     
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out

source venv/bin/activate
cd /home/user

python /home/user/scripts/infer_all_front_back.py \
  --model-name resnet18 \
  --checkpoint /home/user/runs/resnet18_expt/best_model.pth \
  --image-list /home/user/filtered_landscape_images.txt \
  --img-root /data/projects/ml2024/ \
  --output /home/user/back_of_car_filtered.txt \
  --device cuda
```

Submit and monitor:

```
sbatch run_infer_all_front_back.sbatch
tail -f back_of_car_infer_<JOB_ID>.out
wc -l /home/user/back_of_car_filtered.txt
```

**Filtered images:** `1,385,302`

---

## OpenAI API Classifier

1. Install dependencies

2. Make sure OpenAI API key is setup

3. Make sure you have:

   * The **image dataset** in `sampled_images/`
   * The **hand-labeled file**: `ADL/hand_labeled_annotations.json`
   * Double check local directories in scripts before running

#### **Run**

Run openai_test.ipynb

It will:

* Randomly select 100 images by label type from the hand-labeled data
* Call the OpenAI model 
* Output JSON results to:

  ```
  openai_light_predictions.json
  ```

| Model & Prompt        | Turn Signal Accuracy | Tail Light Accuracy |
|----------------------|-------------------|------------------|
| gpt-4o-mini & prompt 1 | 44%               | 41%              |
| gpt-4o & prompt 1      | 63%               | 44%              |
| gpt-4o-mini & prompt 2 | 45%               | 39%              |
| gpt-4o & prompt 2      | 65%               | 52%              |
| gpt-4o-mini & prompt 3 | 40%               | 45%              |
| gpt-4o & prompt 3      | 66%               | 50%              |


---

## DeepSort Crop Sequencing/Tracking

In “/gpfs/space/projects/ml2024/image/predict/labels/xxxxxx.txt” there are the crop “coordinates”, confidence score, etc. Each file is named using a 6-digit number that reflects the order in which it was created. An optional 7th digit is used when an image contains multiple “cars.” This additional digit does not represent a consistent identifier across images. They relate to the jpg crop names in the following:

- Line 0 in txt → crop with just frame ID (e.g., 153781.jpg)
- Line 1 in txt → crop with frame ID + 2 (e.g., 1537812.jpg)
- Line 2 in txt → crop with frame ID + 3 (e.g., 1537813.jpg)


**Files:**
- Input: `/gpfs/helios/home/dwenger/detections_with_crop_path.csv`
- Output: `/gpfs/helios/home/dwenger/tracks_deepsort.csv`
- Script: `/gpfs/helios/home/dwenger/scripts/run_deepsort_tracking.py`

| Column      | Example Value                                    | Meaning / Notes                                                                                     |
|------------|-------------------------------------------------|---------------------------------------------------------------------------------------------------|
| sequence   | 2024-03-25-15-40-16_mapping_tartu/camera_fl    | The video/sequence ID from folder structure.                                                       |
| frame_id   | 000124                                         | Frame number within the sequence.                                                                 |
| img_path   | /gpfs/space/projects/ml2024/.../000124.jpg     | Path to the image file corresponding to this frame. Matches frame ID.                             |
| track_id   | 1, 2, 3, …                                     | Unique tracker ID assigned by ByteTrack. IDs are sequential and consistent within the frame.    |
| class_id   | 2 (mostly), 7 (last row)                       | Object class. This comes from detection CSV.                                                     |
| score      | 0.899246, 0.870372, …                          | Confidence score from the detection.                                                      |
| x1, y1, x2, y2 | 1788,1285,2063,1540, etc.                  | Bounding box coordinates in pixels. They are consistent with the image width/height (width=2064, height=1544). |
| width, height | 2064,1544                                     | Resolution of the frame. Matches sequence.                                                      |
| crop_path | /gpfs/space/projects/ml2024/.../0001243.jpg     | Full path of the crop, rather than just the frame.                                                  |


1. Install Dependencies

```
source venv/bin/activate
pip install deep-sort-realtime opencv-python-headless torch torchvision
```

2. Submit Job

```
cd /gpfs/helios/home/dwenger/scripts
sbatch submit_deepsort.sh
```

3. Monitor

```
# Check status
squeue -u dwenger

# Watch log (Ctrl+C to exit, job continues running)
tail -f deepsort_track_<JOB_ID>.out

# Cancel if needed
scancel <JOB_ID>
```

### Output Format

CSV with columns: `sequence`, `frame_id`, `track_id`, `class_id`, `score`, `x1`, `y1`, `x2`, `y2`, `crop_path`, `img_path`, `width`, `height`

### Parameters to Tune

Edit line 52-62 in `run_deepsort_tracking.py`:

#### `max_cosine_distance` (default: 0.3)
Controls appearance matching strictness (0-1)
- **0.2**: Strict (fewer ID switches, may break tracks)
- **0.3**: Balanced 
- **0.4**: Loose (longer tracks, more potential errors)

#### `max_age` (default: 30)
Frames to keep track alive without detections
- **15**: Drop tracks quickly
- **30**: Balanced 
- **50**: Maintain through occlusions

#### `n_init` (default: 3)
Frames needed to confirm new track
- **2**: Fast initialization
- **3**: Balanced 
- **5**: Very stable

#### `embedder` (default: "mobilenet")
Feature extraction model
- **"mobilenet"**: Fast, good quality 
- **"torchreid"**: Better quality, 2x slower
- **"clip"**: Best quality, 3x slower

### Minimum track length (line 115, default: 3)
```
valid_tracks = track_lengths[track_lengths >= 3].index  # Change number
```

### DeepSORT vs ByteTrack

| | ByteTrack | DeepSORT |
|-|-----------|----------|
| **Speed** | Very fast | Moderate |
| **Basis** | Position only | Appearance + position |
| **ID switches** | More common | Fewer |

Note: ByteTrack was also tested, but DeepSort was used in final sorting of the sequences. 

---

## Final Sequence Review (`scripts/sequence_editor.py`)

After automated filtering, all remaining sequences were manually reviewed using the notebook tool in `scripts/sequence_editor.py`. This script loads the candidate sequences and provides a notebook-based UI for a final, full pass quality check.

During this step, sequences were:
- Visually inspected frame-by-frame
- Relabeled at the frame or sequence level where needed
- Cleaned by removing bad crops
- Split into multiple sequences when temporal boundaries were incorrect
- Merged when multiple sequences represented the same event

The tool supports pagination, undo, and automatic state saving. Once review was complete, the cleaned dataset was exported and is saved under `jupyter/filtered_1010.csv` on the HPC.

| Turn Signal Class | Image Count | Percentage |
|-------------------|-------------|------------|
| None              | 183,964     | 86.4%      |
| Left              | 17,142      | 8.1%       |
| Right             | 11,667      | 5.5%       |
| Hazard            | 146         | 0.07%      |
| **Total**         | **212,919** | **100%**   |

---

## Heuristic Turn Signal Detection (`heuristic_method.py`)

### Method Overview

The heuristic operates on short image sequences and consists of three main stages:

1. **Color-Based Signal Extraction**
   - Frames are converted from RGB to HSV color space.
   - A color mask isolates yellow–orange pixels (hue ≈ 15°–35°) with minimum saturation and brightness thresholds to suppress reflections and background noise.
   - The number of masked pixels per frame yields a scalar “yellow intensity” time series.

2. **Temporal Periodicity Detection**
   - The intensity signal is normalized and analyzed using a Fast Fourier Transform (FFT).
   - Energy in the expected turn-signal frequency band (≈ 1.0–2.5 Hz) is measured.
   - A sequence is considered periodic if the peak spectral power in this band exceeds a fixed multiple of the mean spectrum, filtering out noise and static lights.

3. **Spatial Reasoning for Signal Type**
   - The image is divided into left and right regions of interest (ROIs) corresponding to typical turn signal locations.
   - Intensity time series are computed independently for each ROI.
   - The standard deviation of each series measures blinking activity.
   - Classification logic:
     - No significant periodicity → `none`
     - Similar activity on both sides → `hazard`
     - Higher activity on left or right → `left` or `right`

### Results

When evaluated on the **1010 hand-verified sequences** (212,919 images total), the heuristic method achieved an overall accuracy of approximately **80%**, outperforming the tested foundation-model baselines in this setting.

Per-class accuracy:
- `none`: 0.83  
- `hazard`: 0.76  
- `left`: 0.57  
- `right`: 0.61  

---

## Experimental: Cosmos-Based Labeling Tests (Not Working Yet)

In addition to manual review, I explored using NVIDIA Cosmos (`nvidia/Cosmos-Reason1-7B`) to test whether large multimodal models could assist with turn-signal labeling.

Two experimental scripts were used:

- **Single-frame classification (`scripts/test_cosmos_sequence.py`)**: A small subset of frames from a single sequence was passed to Cosmos with a direct image–text prompt asking for a `left / right / hazard / none` classification.
- **Temporal context classification (`scripts/cosmos_method.py`)**: A sliding window of frames around each target image was provided to Cosmos to evaluate whether short-term temporal context improves signal classification consistency.

**Cosmos Setup**

Running the Cosmos experiments requires a CUDA-enabled GPU environment and a recent version of Hugging Face Transformers with support for multimodal text–image models.

At a minimum, the setup involves:
- A GPU with sufficient VRAM to load `Cosmos-Reason1-7B` (tested with `device_map="auto"` and `bfloat16`)
- PyTorch built with CUDA support
- `transformers` with `trust_remote_code=True` enabled
- `Pillow`, `pandas`, and `tqdm` for data loading and batch processing

Both approaches were implemented as offline batch inference jobs that load crops from the CSV, construct multimodal prompts, and write predictions back to a CSV for later analysis.

**Status / Results:**
- Single-frame Cosmos labeling: **not working yet**
- Temporal-context Cosmos labeling: **not working yet**

At the time of writing, these experiments were exploratory only and did not produce reliable or usable labels. All final annotations therefore rely exclusively on the manual sequence review performed with `scripts/sequence_editor.py`.

---

## Grounding DINO Classifier (this did not work well)

#### **Setup**

1. Install dependencies:

   ```
   pip install groundingdino-py transformers torch torchvision matplotlib
   ```

   *(On M-series Mac, PyTorch will use `mps` automatically.)*

2. Download the model weights:

   ```
   mkdir -p models
   wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth -O models/groundingdino_swint_ogc.pth
   ```

3. Make sure `run_grounding_dino_local.py` points to:

   ```
   IMAGE_LIST_PATH = "sample_100_for_openai.json"  # same images as OpenAI test
   OUTPUT_PATH = "grounding_dino_results.json"
   ```

#### **Run**

```
python run_grounding_dino_local.py
```

It will:

* Load the same 100 images
* Run detection for phrases like `"left turn signal"`, `"right turn signal"`, `"tail light on"`, etc.
* Output structured results to:

  ```
  grounding_dino_results.json
  ```
