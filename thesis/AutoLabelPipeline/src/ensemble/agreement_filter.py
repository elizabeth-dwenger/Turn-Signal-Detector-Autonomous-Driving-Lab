"""
Agreement-based filtering module for two-model ensemble.

Compares predictions from two models (e.g., cosmos_reason2_video and heuristic_model)
on a per-frame basis. Keeps only frames where both models agree on the label;
filters out all frames where they disagree.

Produces:
  - Filtered predictions in the same format as the existing ensemble output
  - Detailed filtering statistics (counts, percentages, common disagreements)
"""
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ensemble.loader import EnsembleLoader, FramePrediction, FramePredictionDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FilteringReport:
    """
    Comprehensive report on what was kept and what was filtered.

    Attributes:
        total_frames: Total frames evaluated (present in both models).
        agreed_frames: Frames where both models gave the same label.
        disagreed_frames: Frames where labels differed.
        missing_model_a_only: Frames present only in model A (no prediction from B).
        missing_model_b_only: Frames present only in model B (no prediction from A).
        agreement_rate: agreed / total  (0-1).
        disagreement_rate: disagreed / total  (0-1).
        agreed_label_distribution: Counter of labels in the *kept* set.
        disagreed_label_pairs: Counter of (model_a_label, model_b_label) in *filtered* set.
        per_label_agreement: For each label, fraction of frames where both models agree.
        per_sequence_stats: Per-sequence agreement rates.
    """
    total_frames: int = 0
    agreed_frames: int = 0
    disagreed_frames: int = 0
    missing_model_a_only: int = 0
    missing_model_b_only: int = 0
    agreement_rate: float = 0.0
    disagreement_rate: float = 0.0
    agreed_label_distribution: Dict[str, int] = field(default_factory=dict)
    disagreed_label_pairs: Dict[str, int] = field(default_factory=dict)
    per_label_agreement: Dict[str, float] = field(default_factory=dict)
    per_sequence_stats: Dict[str, Dict] = field(default_factory=dict)
    model_a_name: str = ""
    model_b_name: str = ""
    timestamp: str = ""

    # ---- helpers ----
    def to_dict(self) -> Dict:
        """Serialise the report to a plain dict (JSON-safe)."""
        return {
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "timestamp": self.timestamp,
            "total_frames": self.total_frames,
            "agreed_frames": self.agreed_frames,
            "disagreed_frames": self.disagreed_frames,
            "missing_model_a_only": self.missing_model_a_only,
            "missing_model_b_only": self.missing_model_b_only,
            "agreement_rate": round(self.agreement_rate, 4),
            "disagreement_rate": round(self.disagreement_rate, 4),
            "agreed_label_distribution": dict(self.agreed_label_distribution),
            "disagreed_label_pairs": dict(self.disagreed_label_pairs),
            "per_label_agreement": {k: round(v, 4) for k, v in self.per_label_agreement.items()},
            "per_sequence_stats": dict(self.per_sequence_stats),
        }

    def summary_text(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 72,
            "AGREEMENT-BASED FILTERING REPORT",
            f"  Model A : {self.model_a_name}",
            f"  Model B : {self.model_b_name}",
            f"  Generated: {self.timestamp}",
            "=" * 72,
            "",
            f"  Total comparable frames : {self.total_frames}",
            f"  Agreed (kept)           : {self.agreed_frames}  ({self.agreement_rate:.1%})",
            f"  Disagreed (filtered)    : {self.disagreed_frames}  ({self.disagreement_rate:.1%})",
            f"  Missing from A only     : {self.missing_model_a_only}",
            f"  Missing from B only     : {self.missing_model_b_only}",
            "",
            "  Kept-label distribution:",
        ]
        for label, count in sorted(self.agreed_label_distribution.items(), key=lambda x: -x[1]):
            pct = count / max(self.agreed_frames, 1) * 100
            lines.append(f"    {label:<12} {count:>6}  ({pct:>5.1f}%)")

        lines.append("")
        lines.append("  Most common disagreement pairs (A → B):")
        top_pairs = sorted(self.disagreed_label_pairs.items(), key=lambda x: -x[1])[:10]
        for pair_str, count in top_pairs:
            pct = count / max(self.disagreed_frames, 1) * 100
            lines.append(f"    {pair_str:<25} {count:>6}  ({pct:>5.1f}%)")

        lines.append("")
        lines.append("  Per-label agreement rate:")
        for label, rate in sorted(self.per_label_agreement.items()):
            lines.append(f"    {label:<12} {rate:.1%}")

        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heuristic result loader
# ---------------------------------------------------------------------------

class HeuristicResultLoader:
    """
    Load heuristic model predictions from the results.json produced by
    ``heuristic_method/test_heuristic.py``.

    Converts the per-sequence segment predictions into a
    ``FramePredictionDataset`` compatible with the ensemble pipeline.
    """

    LABEL_MAP = {"both": "hazard", "signal": "hazard", "multiple": "hazard"}
    VALID_LABELS = {"left", "right", "hazard", "none"}

    def __init__(self, model_name: str = "heuristic_model"):
        self.model_name = model_name

    @staticmethod
    def _harmonize_label(label: str) -> str:
        if label is None:
            return "none"
        label = str(label).strip().lower()
        label = HeuristicResultLoader.LABEL_MAP.get(label, label)
        if label not in HeuristicResultLoader.VALID_LABELS:
            logger.warning(f"Invalid heuristic label '{label}'; defaulting to 'none'")
            return "none"
        return label

    def load(self, results_path: Path) -> FramePredictionDataset:
        """
        Load heuristic predictions from the JSON results file.

        Args:
            results_path: Path to ``results.json`` produced by test_heuristic.py,
                          OR a directory containing ``results.json``.

        Returns:
            FramePredictionDataset populated with per-frame predictions.
        """
        results_path = Path(results_path)

        # Accept either a file or directory
        if results_path.is_dir():
            json_file = results_path / "results.json"
            if not json_file.exists():
                # Try to find any results.json recursively
                candidates = list(results_path.rglob("results.json"))
                if candidates:
                    json_file = candidates[0]
                    logger.info(f"Found heuristic results at {json_file}")
                else:
                    raise FileNotFoundError(
                        f"No results.json found in {results_path}"
                    )
        else:
            json_file = results_path

        if not json_file.exists():
            raise FileNotFoundError(f"Heuristic results file not found: {json_file}")

        with open(json_file) as f:
            data = json.load(f)

        dataset = FramePredictionDataset()
        dataset.metadata[self.model_name] = None  # placeholder

        results_list = data.get("results", [])
        if not results_list:
            logger.warning(f"No results entries in {json_file}")
            return dataset

        loaded_frames = 0
        for entry in results_list:
            sequence_id = entry.get("sequence_id", "")
            prediction = entry.get("prediction", {})
            segments = prediction.get("segments", [])
            overall_label = self._harmonize_label(prediction.get("label", "none"))
            num_frames = entry.get("num_frames", 0)

            if segments:
                # Expand segments into per-frame predictions
                for seg in segments:
                    seg_label = self._harmonize_label(seg.get("label", overall_label))
                    start = seg.get("start_frame", 0)
                    end = seg.get("end_frame", num_frames - 1)
                    for frame_id in range(start, end + 1):
                        pred = FramePrediction(
                            frame_id=frame_id,
                            label=seg_label,
                            confidence=0.5,  # heuristic has no confidence
                            raw_output="heuristic",
                            model_name=self.model_name,
                        )
                        key = (sequence_id, frame_id)
                        if key not in dataset.predictions:
                            dataset.predictions[key] = []
                        dataset.predictions[key].append(pred)
                        loaded_frames += 1
            else:
                # Single label for entire sequence — expand if num_frames known
                n = max(num_frames, 1)
                for frame_id in range(n):
                    pred = FramePrediction(
                        frame_id=frame_id,
                        label=overall_label,
                        confidence=0.5,
                        raw_output="heuristic",
                        model_name=self.model_name,
                    )
                    key = (sequence_id, frame_id)
                    if key not in dataset.predictions:
                        dataset.predictions[key] = []
                    dataset.predictions[key].append(pred)
                    loaded_frames += 1

        logger.info(
            f"Loaded {loaded_frames} heuristic frame predictions "
            f"across {len(set(k[0] for k in dataset.predictions))} sequences"
        )
        return dataset


# ---------------------------------------------------------------------------
# VLM result loader (thin wrapper around EnsembleLoader)
# ---------------------------------------------------------------------------

class VLMResultLoader:
    """
    Load VLM model predictions using the existing ``EnsembleLoader``.

    Handles path discovery through the ``results/<model>/test_runs/<ts>/``
    directory structure.
    """

    def __init__(self, model_name: str = "cosmos_reason2_video",
                 transform_sequence_ids: bool = True):
        self.model_name = model_name
        self.loader = EnsembleLoader(
            label_harmonization=True,
            calibrate_confidence=False,
            handle_missing_frames="exclude",
            transform_sequence_ids=transform_sequence_ids,
        )

    def load(self, results_dir: Path) -> FramePredictionDataset:
        """
        Load VLM predictions from the standard results directory.

        Args:
            results_dir: Either the direct directory containing CSV/JSON files,
                         or the model root (e.g., ``results/cosmos_reason2_video``)
                         in which case ``test_runs/<latest>/`` is auto-discovered.

        Returns:
            FramePredictionDataset with per-frame predictions.
        """
        results_dir = Path(results_dir)

        # Auto-discover test_runs/<timestamp>/ if needed
        test_runs_dir = results_dir / "test_runs"
        if test_runs_dir.exists():
            timestamp_dirs = sorted(
                [d for d in test_runs_dir.iterdir() if d.is_dir()],
                key=lambda p: p.name, reverse=True,
            )
            if timestamp_dirs:
                results_dir = timestamp_dirs[0]
                logger.info(f"Auto-discovered VLM results dir: {results_dir}")

        return self.loader.load_from_results_dir(
            results_dir,
            model_name=self.model_name,
            model_config={"mode": "video"},
        )


# ---------------------------------------------------------------------------
# Core agreement filter
# ---------------------------------------------------------------------------

class AgreementFilter:
    """
    Compare predictions from two models on a per-frame basis and keep only
    frames where both models agree on the label.

    Usage::

        af = AgreementFilter(model_a_name="cosmos_reason2_video",
                             model_b_name="heuristic_model")
        filtered_df, report = af.filter(dataset_a, dataset_b)
    """

    VALID_LABELS = {"left", "right", "hazard", "none"}

    def __init__(
        self,
        model_a_name: str = "cosmos_reason2_video",
        model_b_name: str = "heuristic_model",
    ):
        """
        Args:
            model_a_name: Human-readable identifier for the first model.
            model_b_name: Human-readable identifier for the second model.
        """
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def filter(
        self,
        dataset_a: FramePredictionDataset,
        dataset_b: FramePredictionDataset,
    ) -> Tuple[pd.DataFrame, FilteringReport]:
        """
        Run agreement-based filtering.

        For each frame key ``(sequence_id, frame_id)`` that exists in *both*
        datasets, compare the predicted labels.  Keep the frame (using
        model A's prediction) if the labels match; filter it out otherwise.

        Args:
            dataset_a: Predictions from the first model (e.g., cosmos VLM).
            dataset_b: Predictions from the second model (e.g., heuristic).

        Returns:
            filtered_df: DataFrame in the same schema as ensemble output
                (sequence_id, frame_id, ensemble_label, ensemble_confidence,
                 agreement_count, total_models, voting_distribution).
            report: A ``FilteringReport`` with all statistics.
        """
        keys_a = set(dataset_a.predictions.keys())
        keys_b = set(dataset_b.predictions.keys())
        common_keys = keys_a & keys_b
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a

        logger.info(
            f"Agreement filter: {len(keys_a)} frames from {self.model_a_name}, "
            f"{len(keys_b)} from {self.model_b_name}, "
            f"{len(common_keys)} in common"
        )

        if not common_keys:
            logger.warning(
                "No overlapping frame keys between the two models. "
                "Check that sequence_id and frame_id formats match."
            )
            # Log sample keys for diagnosis
            sample_a = list(keys_a)[:3]
            sample_b = list(keys_b)[:3]
            logger.warning(f"  Sample keys A: {sample_a}")
            logger.warning(f"  Sample keys B: {sample_b}")

        # ----- per-frame comparison -----
        agreed_rows: List[Dict] = []
        disagreed_rows: List[Dict] = []

        agreed_labels = Counter()
        disagreed_pairs = Counter()
        per_label_total = Counter()      # total frames where A predicts label L
        per_label_agreed = Counter()     # … and B agrees

        per_seq_total = defaultdict(int)
        per_seq_agreed = defaultdict(int)

        for key in sorted(common_keys):
            seq_id, frame_id = key
            preds_a = dataset_a.predictions[key]
            preds_b = dataset_b.predictions[key]

            # Use first prediction per model
            label_a = preds_a[0].label if preds_a else "none"
            conf_a = preds_a[0].confidence if preds_a else 0.0
            label_b = preds_b[0].label if preds_b else "none"
            conf_b = preds_b[0].confidence if preds_b else 0.0

            per_label_total[label_a] += 1
            per_seq_total[seq_id] += 1

            if label_a == label_b:
                agreed_labels[label_a] += 1
                per_label_agreed[label_a] += 1
                per_seq_agreed[seq_id] += 1

                agreed_rows.append({
                    "sequence_id": seq_id,
                    "frame_id": frame_id,
                    "ensemble_label": label_a,
                    "ensemble_confidence": round((conf_a + conf_b) / 2, 4),
                    "agreement_count": 2,
                    "total_models": 2,
                    "voting_distribution": json.dumps({label_a: 2}),
                    "model_a_label": label_a,
                    "model_b_label": label_b,
                })
            else:
                pair_key = f"{label_a} -> {label_b}"
                disagreed_pairs[pair_key] += 1

                disagreed_rows.append({
                    "sequence_id": seq_id,
                    "frame_id": frame_id,
                    "model_a_label": label_a,
                    "model_a_confidence": conf_a,
                    "model_b_label": label_b,
                    "model_b_confidence": conf_b,
                })

        # ----- build report -----
        total = len(common_keys)
        agreed = len(agreed_rows)
        disagreed = len(disagreed_rows)

        per_label_agree_rate = {}
        for label in self.VALID_LABELS:
            denom = per_label_total.get(label, 0)
            if denom > 0:
                per_label_agree_rate[label] = per_label_agreed.get(label, 0) / denom
            else:
                per_label_agree_rate[label] = 0.0

        per_seq_stats = {}
        for seq_id in per_seq_total:
            t = per_seq_total[seq_id]
            a = per_seq_agreed.get(seq_id, 0)
            per_seq_stats[seq_id] = {
                "total": t,
                "agreed": a,
                "disagreed": t - a,
                "agreement_rate": round(a / t, 4) if t else 0.0,
            }

        report = FilteringReport(
            total_frames=total,
            agreed_frames=agreed,
            disagreed_frames=disagreed,
            missing_model_a_only=len(only_a),
            missing_model_b_only=len(only_b),
            agreement_rate=agreed / total if total else 0.0,
            disagreement_rate=disagreed / total if total else 0.0,
            agreed_label_distribution=dict(agreed_labels),
            disagreed_label_pairs=dict(disagreed_pairs),
            per_label_agreement=per_label_agree_rate,
            per_sequence_stats=per_seq_stats,
            model_a_name=self.model_a_name,
            model_b_name=self.model_b_name,
            timestamp=datetime.now().isoformat(),
        )

        # ----- build output DataFrame (same schema as ensemble_predictions.csv) -----
        if agreed_rows:
            filtered_df = pd.DataFrame(agreed_rows)
        else:
            filtered_df = pd.DataFrame(
                columns=[
                    "sequence_id", "frame_id", "ensemble_label",
                    "ensemble_confidence", "agreement_count", "total_models",
                    "voting_distribution", "model_a_label", "model_b_label",
                ]
            )

        logger.info(
            f"Agreement filtering complete: "
            f"{agreed}/{total} frames kept ({report.agreement_rate:.1%}), "
            f"{disagreed} filtered out"
        )

        return filtered_df, report

    # ------------------------------------------------------------------ #
    # Convenience: save outputs
    # ------------------------------------------------------------------ #

    def save_results(
        self,
        filtered_df: pd.DataFrame,
        report: FilteringReport,
        output_dir: Path,
        disagreed_rows: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Path]:
        """
        Persist filtered predictions, filtering report, and disagreement log.

        Args:
            filtered_df: Kept-prediction DataFrame from ``filter()``.
            report: ``FilteringReport`` from ``filter()``.
            output_dir: Directory to write into (created if needed).
            disagreed_rows: Optional DataFrame of disagreed frames.

        Returns:
            Dict mapping output name to file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}

        # 1. Filtered predictions CSV (same format as ensemble_predictions.csv)
        pred_path = output_dir / "agreement_predictions.csv"
        # Drop helper columns before saving to match ensemble format
        export_cols = [
            "sequence_id", "frame_id", "ensemble_label",
            "ensemble_confidence", "agreement_count", "total_models",
            "voting_distribution",
        ]
        existing_cols = [c for c in export_cols if c in filtered_df.columns]
        filtered_df[existing_cols].to_csv(pred_path, index=False)
        paths["predictions"] = pred_path
        logger.info(f"Saved agreement predictions to {pred_path}")

        # 2. Full report JSON
        report_path = output_dir / "agreement_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        paths["report"] = report_path

        # 3. Human-readable summary
        summary_path = output_dir / "agreement_summary.txt"
        with open(summary_path, "w") as f:
            f.write(report.summary_text())
        paths["summary"] = summary_path

        # 4. Disagreement details CSV (if provided)
        if disagreed_rows is not None and not disagreed_rows.empty:
            dis_path = output_dir / "disagreed_frames.csv"
            disagreed_rows.to_csv(dis_path, index=False)
            paths["disagreements"] = dis_path
            logger.info(f"Saved {len(disagreed_rows)} disagreed frames to {dis_path}")

        # 5. Metadata (for reproducibility)
        meta = {
            "model_a": report.model_a_name,
            "model_b": report.model_b_name,
            "timestamp": report.timestamp,
            "total_frames": report.total_frames,
            "agreed_frames": report.agreed_frames,
            "disagreed_frames": report.disagreed_frames,
            "agreement_rate": report.agreement_rate,
            "output_files": {k: str(v) for k, v in paths.items()},
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        paths["metadata"] = meta_path

        return paths

    # ------------------------------------------------------------------ #
    # Full convenience pipeline
    # ------------------------------------------------------------------ #

    def run(
        self,
        dataset_a: FramePredictionDataset,
        dataset_b: FramePredictionDataset,
        output_dir: Path,
        ground_truth: Optional[Dict[Tuple[str, int], str]] = None,
    ) -> Dict:
        """
        End-to-end: filter, evaluate (optional), save everything.

        Args:
            dataset_a: Predictions from model A.
            dataset_b: Predictions from model B.
            output_dir: Where to write results.
            ground_truth: Optional mapping (sequence_id, frame_id) → label.

        Returns:
            Dict with keys: filtered_df, report, metrics (if ground_truth given),
            and output_paths.
        """
        filtered_df, report = self.filter(dataset_a, dataset_b)

        # Build disagreed DataFrame for saving
        keys_a = set(dataset_a.predictions.keys())
        keys_b = set(dataset_b.predictions.keys())
        common_keys = keys_a & keys_b
        disagreed_rows_list = []
        for key in sorted(common_keys):
            preds_a = dataset_a.predictions[key]
            preds_b = dataset_b.predictions[key]
            la = preds_a[0].label if preds_a else "none"
            lb = preds_b[0].label if preds_b else "none"
            if la != lb:
                disagreed_rows_list.append({
                    "sequence_id": key[0],
                    "frame_id": key[1],
                    "model_a_label": la,
                    "model_a_confidence": preds_a[0].confidence if preds_a else 0.0,
                    "model_b_label": lb,
                    "model_b_confidence": preds_b[0].confidence if preds_b else 0.0,
                })
        disagreed_df = pd.DataFrame(disagreed_rows_list) if disagreed_rows_list else pd.DataFrame()

        # Save
        output_paths = self.save_results(
            filtered_df, report, output_dir, disagreed_rows=disagreed_df,
        )

        # Evaluate against ground truth if provided
        metrics = {}
        if ground_truth and not filtered_df.empty:
            from ensemble.evaluator import EnsembleEvaluator
            evaluator = EnsembleEvaluator()

            y_true, y_pred = [], []
            for _, row in filtered_df.iterrows():
                key = (row["sequence_id"], int(row["frame_id"]))
                if key in ground_truth:
                    y_true.append(ground_truth[key])
                    y_pred.append(row["ensemble_label"])

            if y_true:
                metrics = evaluator.compute_frame_metrics(y_true, y_pred, per_label=True)
                logger.info(
                    f"Evaluation on agreed frames: "
                    f"macro_f1={metrics['frame_macro_f1']:.3f}, "
                    f"accuracy={metrics['accuracy']:.3f} "
                    f"({len(y_true)} frames with GT)"
                )
                # Save metrics
                metrics_path = Path(output_dir) / "evaluation_metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2, default=str)
                output_paths["metrics"] = metrics_path
            else:
                logger.warning("No matching ground-truth keys for the agreed frames")

        # Print summary
        print("\n" + report.summary_text())

        return {
            "filtered_df": filtered_df,
            "disagreed_df": disagreed_df,
            "report": report,
            "metrics": metrics,
            "output_paths": output_paths,
        }
