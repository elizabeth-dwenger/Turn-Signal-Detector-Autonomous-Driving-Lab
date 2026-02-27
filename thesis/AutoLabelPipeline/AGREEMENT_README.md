# Agreement-Based Ensemble Filtering

Two-model agreement filter that keeps **only** the frames where both `cosmos_reason2_video` and `heuristic_model` predict the same label. All disagreements are filtered out and logged for analysis.

---

## Overview

| Step | What happens |
|------|-------------|
| 1    | Load per-frame predictions from `cosmos_reason2_video` (VLM) |
| 2    | Load per-frame predictions from `heuristic_model` |
| 3    | Match frames by `(sequence_id, frame_id)` |
| 4    | **Keep** frame if both labels agree; **discard** otherwise |
| 5    | Save filtered predictions + detailed filtering report |
| 6    | Optionally evaluate against ground truth |

---

## File Layout

```
src/ensemble/agreement_filter.py   # Core module (AgreementFilter, loaders, report)
run_agreement_ensemble.py          # CLI entry point
configs/agreement_ensemble_config.yaml  # Config template
```

---

## Step-by-Step Usage

### 1. Generate predictions from `cosmos_reason2_video`

Run the VLM pipeline with the cosmos config. This produces per-sequence CSV/JSON
files under `results/cosmos_reason2_video/test_runs/<timestamp>/`.

```
python run_pipeline.py --config configs/cosmos_reason2_video.yaml
```

Output structure:
```
results/cosmos_reason2_video/
└── test_runs/
    └── cosmos_20260208_120000/
        ├── <sequence_id>__track_<track>.csv
        ├── <sequence_id>__track_<track>.json
        └── ...
```

### 2. Generate predictions from `heuristic_model`

Run the heuristic detector pipeline. This produces a single `results.json`
inside `prompt_comparison/heuristic_<timestamp>/`.

```
python -m heuristic_method.test_heuristic \
    --config configs/cosmos_reason2_video.yaml \
    --test-sequences data/test_sequences.json \
    --output-dir prompt_comparison \
    --fps 5.0 \
    --activity-threshold 1000.0 \
    --verbose
```

Output structure:
```
prompt_comparison/
└── heuristic_20260215_120000/
    └── results.json
```

### 3. Run the agreement-based ensemble

#### Option A — CLI flags

```
python run_agreement_ensemble.py \
    --vlm-results results/cosmos_reason2_video \
    --heuristic-results prompt_comparison/heuristic_20260215_120000/results.json \
    --ground-truth data/tracking_data.csv \
    --output-dir agreement_experiments/ \
    --verbose
```

#### Option B — Config file

Edit `configs/agreement_ensemble_config.yaml` to point to your actual paths,
then:

```bash
python run_agreement_ensemble.py \
    --config configs/agreement_ensemble_config.yaml
```

### 4. Locate and interpret results

After running, the output directory looks like:

```
agreement_experiments/
└── agreement_20260225_143000/
    ├── agreement_predictions.csv   # Kept frames (same schema as ensemble output)
    ├── agreement_report.json       # Full filtering statistics (JSON)
    ├── agreement_summary.txt       # Human-readable summary
    ├── disagreed_frames.csv        # Filtered-out frames with both labels
    ├── evaluation_metrics.json     # Metrics vs ground truth (if provided)
    └── metadata.json               # Run metadata
```

#### Key files

| File | Description |
|------|-------------|
| `agreement_predictions.csv` | Frames where both models agreed — same columns as `ensemble_predictions.csv` (`sequence_id, frame_id, ensemble_label, ensemble_confidence, agreement_count, total_models, voting_distribution`) |
| `disagreed_frames.csv` | Frames where models disagreed — columns: `sequence_id, frame_id, model_a_label, model_a_confidence, model_b_label, model_b_confidence` |
| `agreement_report.json` | Overall stats: total/agreed/disagreed counts, rates, per-label breakdown, per-sequence stats |
| `agreement_summary.txt` | Same info as report but formatted for quick reading |
| `evaluation_metrics.json` | Frame-level metrics (macro F1, per-class F1, confusion matrix) computed on the agreed frames only |

---

## Understanding the Report

The `agreement_summary.txt` includes:

- **Agreement / disagreement counts and rates** — how many frames survived filtering
- **Kept-label distribution** — label breakdown of the agreed set
- **Most common disagreement pairs** — e.g., `"none -> left"` means the VLM predicted `none` but the heuristic predicted `left`
- **Per-label agreement rate** — for each label, what fraction of frames with that label in model A were also predicted by model B
- **Per-sequence stats** — agreement rate per video sequence

---

## Programmatic Usage

```
import sys
from pathlib import Path

sys.path.insert(0, "src")

from ensemble.agreement_filter import (
    AgreementFilter,
    HeuristicResultLoader,
    VLMResultLoader,
)

# Load predictions
vlm = VLMResultLoader("cosmos_reason2_video").load(Path("results/cosmos_reason2_video"))
heur = HeuristicResultLoader("heuristic_model").load(Path("prompt_comparison/heuristic_<timestamp>"))

# Filter
af = AgreementFilter(model_a_name="cosmos_reason2_video",
                     model_b_name="heuristic_model")
filtered_df, report = af.filter(vlm, heur)

# Inspect
print(report.summary_text())
print(f"Kept {len(filtered_df)} / {report.total_frames} frames")

# Save everything (including disagreement log)
af.run(vlm, heur, output_dir=Path("agreement_experiments/my_run"))
```

---

## Edge Cases Handled

| Scenario | Behaviour |
|----------|-----------|
| Frame in model A but not B | Counted as `missing_model_b_only`; **not included** in output |
| Frame in model B but not A | Counted as `missing_model_a_only`; **not included** in output |
| Zero overlapping frames | Warning logged with sample keys for debugging; empty output |
| Different label formats | Labels are harmonized (`"both"` → `"hazard"`, invalid → `"none"`) |
| Multiple predictions per frame per model | First prediction used (list position 0) |
| No ground truth provided | Evaluation skipped; predictions and report still saved |