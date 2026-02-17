#!/usr/bin/env python
"""
Summarize pipeline runs into a comparison table without re-running inference.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd


def _parse_timestamp(ts: str) -> datetime:
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except Exception:
        return datetime.min


def _infer_inference_mode(model: str, inference_mode: str = None, run_dir: Path = None) -> str:
    if inference_mode:
        mode = str(inference_mode).strip().lower()
        if mode in {"video", "single", "single_image", "image", "single-image"}:
            return "video" if mode == "video" else "single"
    model_l = str(model).lower() if model else ""
    if "video" in model_l:
        return "video"
    if "single" in model_l or "image" in model_l:
        return "single"
    if run_dir is not None:
        name = str(run_dir).lower()
        if "video" in name:
            return "video"
        if "single" in name or "image" in name:
            return "single"
    return "unknown"


def _model_display_name(model: str, inference_mode: str) -> str:
    suffix = inference_mode if inference_mode else "unknown"
    return f"{model}_{suffix}"


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def summarize_runs(root_dir: Path) -> None:
    eval_files = list(root_dir.glob("**/evaluation_metrics.json"))
    if not eval_files:
        print(f"No evaluation_metrics.json files found under {root_dir}")
        return

    rows = []
    for eval_path in sorted(eval_files):
        run_dir = eval_path.parent
        eval_data = _load_json(eval_path)
        model_metrics = _load_json(run_dir / "model_metrics.json")

        model = model_metrics.get("model", "unknown_model")
        inference_mode = _infer_inference_mode(model, model_metrics.get("inference_mode"), run_dir)
        model_display = _model_display_name(model, inference_mode)

        timestamp = model_metrics.get("timestamp")
        if not timestamp:
            timestamp = datetime.fromtimestamp(eval_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")

        frame_metrics = eval_data.get("frame_metrics", {})
        event_metrics = eval_data.get("event_metrics", {})
        per_class = frame_metrics.get("per_class", {})

        rows.append({
            "model_display": model_display,
            "timestamp": timestamp,
            "timestamp_dt": _parse_timestamp(timestamp),
            "sequences": eval_data.get("num_sequences", None),
            "frame_accuracy": frame_metrics.get("accuracy", None),
            "frame_macro_f1": frame_metrics.get("macro_f1", None),
            "event_f1": event_metrics.get("f1", None),
            "f1_left": per_class.get("left", {}).get("f1", None) if per_class else None,
            "f1_right": per_class.get("right", {}).get("f1", None) if per_class else None,
            "f1_hazard": per_class.get("hazard", {}).get("f1", None) if per_class else None,
            "f1_none": per_class.get("none", {}).get("f1", None) if per_class else None,
            "avg_latency_ms": model_metrics.get("avg_latency_ms", None)
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No usable data found.")
        return

    # Assign exp ids per model based on timestamp order (ascending)
    df = df.sort_values(["model_display", "timestamp_dt", "timestamp"], ascending=[True, True, True])
    df["exp_id"] = df.groupby("model_display").cumcount().map(lambda i: f"exp_{i}")

    # Order models by best frame accuracy (desc)
    def _acc_value(val):
        return val if isinstance(val, (int, float)) else -1.0

    best_acc = df.groupby("model_display")["frame_accuracy"].apply(
        lambda s: max((_acc_value(v) for v in s), default=-1.0)
    )
    model_order = best_acc.sort_values(ascending=False).index.tolist()
    df["model_order"] = df["model_display"].apply(
        lambda m: model_order.index(m) if m in model_order else len(model_order)
    )
    df["accuracy_sort"] = df["frame_accuracy"].apply(_acc_value)

    df = df.sort_values(
        ["model_order", "accuracy_sort", "timestamp_dt", "timestamp"],
        ascending=[True, False, True, True]
    )

    table = df.set_index(["model_display", "exp_id"])[
        [
            "timestamp",
            "sequences",
            "frame_accuracy",
            "frame_macro_f1",
            "event_f1",
            "f1_left",
            "f1_right",
            "f1_hazard",
            "f1_none",
            "avg_latency_ms",
        ]
    ]

    display_table = table.rename(columns={
        "timestamp": "Timestamp",
        "sequences": "Sequences",
        "frame_accuracy": "Frame Accuracy",
        "frame_macro_f1": "Frame Macro F1",
        "event_f1": "Event F1",
        "f1_left": "F1 Left",
        "f1_right": "F1 Right",
        "f1_hazard": "F1 Hazard",
        "f1_none": "F1 None",
        "avg_latency_ms": "Avg Latency",
    })

    print(display_table.to_string())

    output_path = root_dir / "comparison_summary_all.csv"
    display_table.to_csv(output_path)
    print(f"\nComparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize pipeline runs into a comparison table")
    parser.add_argument("--root", type=str, default="results",
                        help="Root directory containing run outputs (default: results/)")
    args = parser.parse_args()

    summarize_runs(Path(args.root))


if __name__ == "__main__":
    main()
