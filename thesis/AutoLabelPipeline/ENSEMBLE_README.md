# Ensemble Prediction Module

Post-hoc ensemble methods for combining predictions from multiple models **without re-running inference**.

## Overview

This module enables rapid experimentation with majority vote ensemble methods on top of already-saved model predictions. Instead of re-running expensive model inference, you can:

1. **Load** predictions from saved model runs (CSV/JSON)
2. **Aggregate** using majority voting
3. **Evaluate** ensemble performance
4. **Analyze** disagreements to understand model behavior

## Architecture

### Core Components

- **`loader.py`**: Load predictions from saved model runs; normalize labels and confidences
- **`aggregator.py`**: Implement majority voting mechanism
- **`evaluator.py`**: Compute metrics (F1, precision, recall, confusion matrix)
- **`disagreement.py`**: Analyze model disagreements; categorize patterns; identify error correlations
- **`experiment_runner.py`**: Orchestrate end-to-end experiments
- **`run_ensemble.py`**: CLI entry point

## Quick Start

### Prerequisites

Ensure your models have saved predictions in one of these formats:

#### Expected Directory Structure

The ensemble module expects results to be organized as:
```
results/
├── qwen3_vl_single/
│   └── test_runs/
│       └── qwen3_vl_20260207_161719/
│           ├── sequence1.csv
│           ├── sequence1.json
│           ├── sequence2.csv
│           └── ...
├── cosmos_reason2_video/
│   └── test_runs/
│       └── cosmos_20260208_120000/
│           └── ...
```

The loader automatically finds the most recent timestamped directory within `test_runs/`.

#### CSV Format
```
frame_id,label,confidence
0,left,0.85
1,none,0.92
...
```

#### JSON Format
```
{
  "metadata": {"sequence_id": "video_001_track_01"},
  "predictions": [
    {"frame_id": 0, "label": "left", "confidence": 0.85},
    {"frame_id": 1, "label": "none", "confidence": 0.92}
  ]
}
```

**Note**: Metadata files like `dataset_summary.json`, `evaluation_metrics.json`, and `*_review_queue.json` are automatically excluded.

### Running the Ensemble

#### Basic Usage (Majority Vote)

```
python run_ensemble.py \
  --results-root results/ \
  --output-dir ensemble_experiments/ \
  --verbose
```

#### With Ground Truth Evaluation

```
python run_ensemble.py \
  --results-root results/ \
  --ground-truth data/ground_truth.csv \
  --output-dir ensemble_experiments/ \
  --verbose
```

#### Custom Model Selection

```
# Specify models manually with F1 scores
python run_ensemble.py \
  --results-root results/ \
  --models "qwen3_vl_single/exp_5:0.574" "cosmos_reason2_video/exp_2:0.465" \
  --output-dir ensemble_experiments/
```

**Model Selection Strategy**:
- By default, uses top 3 models from built-in F1 scores
- Model names like `"qwen3_vl_single/exp_5"` map to `results/qwen3_vl_single/test_runs/<timestamp>/`
- The loader automatically finds the most recent timestamped subdirectory
- Use `--n-top-models` to change the number of models selected
- Use `--models` to explicitly specify which models and their F1 scores

#### Using Configuration File

```
python run_ensemble.py --config configs/ensemble_config_v1.yaml
```

## Voting Mechanism

### Majority Vote (MV)

Simplest approach: label with >50% agreement. When votes are tied, confidence scores are used as a tie-breaker.

```
from ensemble.aggregator import MajorityVoter, EnsembleAggregator

voter = MajorityVoter(tie_break_method="confidence")
aggregator = EnsembleAggregator(voter)
ensemble_df = aggregator.aggregate(frame_predictions)
```

**Pros**: Simple, interpretable, robust  
**Cons**: Doesn't leverage confidence for voting itself (only for tie-breaking)

## Output Files

After running an ensemble experiment, you'll find:

```
ensemble_experiments/
├── exp_mv/                          # Majority vote results
│   ├── ensemble_predictions.csv     # Ensemble predictions (all frames)
│   ├── disagreement_log.json        # High-disagreement frames
│   └── metadata.json                # Experiment metadata & metrics
├── model_comparison.csv             # Individual vs. ensemble comparison
├── summary.txt                      # Human-readable summary
└── ensemble_*.log                   # Detailed execution log
```

### Key Output Columns

**ensemble_predictions.csv**:
- `sequence_id`: Video identifier
- `frame_id`: Frame number
- `ensemble_label`: Predicted label (left/right/hazard/none)
- `ensemble_confidence`: Confidence score [0, 1]
- `agreement_count`: Number of models voting for this label
- `voting_distribution`: JSON dict of votes per label

## Data Assumptions

1. **Saved predictions must exist** for all models being ensembled
2. **Label harmonization**: Maps variant labels to standard set (left/right/hazard/none)
   - "both" → "hazard"
   - Invalid labels → "none"
3. **Sequence ID transformation**: Automatically converts prediction format to match ground truth
   - Prediction format: `2024-07-09-16-49-42_mapping_tartu_streets_camera_wide_right_170__track_170`
   - Ground truth format: `2024-07-09-16-49-42_mapping_tartu_streets/camera_wide_right_170`
   - Transformation extracts date, location, camera, and track_id components
4. **Confidence calibration**: Scores are in [0, 1]; no post-hoc scaling (yet)
5. **Temporal alignment**: Single-image and video models handled automatically
6. **Missing frames**: Handled via `handle_missing_frames="exclude"` (can switch to "abstain")

## Configuration

See `configs/ensemble_config_template.yaml` for full configuration options. Key settings:

```
ensemble_experiment:
  name: "my_ensemble_v1"

model_selection:
  n_top_models: 3  # Select top 3 by F1
  results_root: "results/"

voters_to_test:
  majority_vote:
    enabled: true
    params:
      tie_break_method: "confidence"

output:
  output_dir: "ensemble_experiments/"
```

## Evaluation Metrics

The module computes:

- **Frame-level F1** (macro, unweighted average across 4 classes)
- **Per-label metrics**: Precision, Recall, F1 for each class
- **Confusion matrix**: 4×4 breakdown of predictions
- **Agreement rate**: Average % of models voting for final prediction

## API Reference

### EnsembleLoader

```python
loader = EnsembleLoader(
    label_harmonization=True,
    calibrate_confidence=False,
    handle_missing_frames="exclude",
    transform_sequence_ids=True  # Converts prediction format to ground truth format
)

# Load from directory
dataset = loader.load_from_results_dir(
    results_dir="results/model_a/",
    model_name="model_a",
    model_config={"mode": "single"}
)

# Merge multiple datasets
merged = loader.merge_datasets(dataset1, dataset2, dataset3)

# Load ground truth
loader.load_ground_truth("ground_truth.csv", dataset)

# Get validation report
report = loader.validate_dataset(dataset)
```

### EnsembleAggregator

```
# Create voter
voter = MajorityVoter(tie_break_method="confidence")

# Aggregate
aggregator = EnsembleAggregator(voter)
ensemble_df = aggregator.aggregate(dataset.predictions, verbose=True)

# Output: DataFrame with columns
# - sequence_id, frame_id
# - ensemble_label, ensemble_confidence
# - agreement_count, total_models
# - voting_distribution
```

### EnsembleEvaluator

```
evaluator = EnsembleEvaluator()

# Compute metrics
metrics = evaluator.compute_frame_metrics(y_true, y_pred)

# Compare individual models vs. ensemble
comparison = evaluator.compare_models(
    individual_results={"model_a": df_a, "model_b": df_b},
    ensemble_result=ensemble_df,
    ground_truth=gt_dict
)
```

### DisagreementAnalyzer

```
analyzer = DisagreementAnalyzer()

# Identify high-disagreement frames
disagreements = analyzer.identify_disagreement_frames(
    frame_predictions, entropy_threshold=0.5
)

# Categorize patterns
patterns = analyzer.categorize_patterns(disagreements)

# Analyze error correlations
correlation = analyzer.analyze_error_correlation(
    frame_predictions, ground_truth
)

# Export log
analyzer.export_disagreement_log(disagreements, "log.json")
```

## Troubleshooting

### No predictions loaded
- Check that result directories follow expected structure
- Verify CSV/JSON files are present and readable
- Use `--verbose` flag to see debug logs
- Check that metadata files are being filtered out correctly

### No evaluation metrics computed (empty summary)
**Symptom**: Summary shows "⚠ No evaluation metrics computed"

**Cause**: No matching frames between ensemble predictions and ground truth

**The ensemble loader automatically transforms sequence_id formats**, but if transformation fails:

**Solution**: 
1. Run with `--verbose` to see transformation logs:
   ```
   INFO - Transforming sequence_id: '2024-07-09-...__track_170' -> '2024-07-09-.../camera_..._170'
   ```
2. Check sample keys from debug output to ensure transformation worked
3. Verify your ground truth CSV has columns: `sequence_id, frame_id, true_label` (or `label`)
4. If transformation is incorrect, you can disable it and manually fix formats:
   ```python
   loader = EnsembleLoader(transform_sequence_ids=False)
   ```

**Quick diagnostic**:
```bash
# Check ensemble predictions (after transformation)
head ensemble_experiments/exp_mv/ensemble_predictions.csv

# Compare with ground truth format (columns: sequence_id, frame_id, true_label)
head data/tracking_data.csv | cut -d',' -f14,3,15
```

### Low ensemble performance
- Ensure individual models are diverse in their predictions
- Check that all models are predicting the same label set
- Verify confidence scores are well-calibrated
- Review disagreement analysis to identify systematic error patterns