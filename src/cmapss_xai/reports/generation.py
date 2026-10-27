from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_statistics_report(
    train: pd.DataFrame,
    sensor_report: pd.DataFrame,
    window_metadata: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_lengths = train.groupby("unit_id")["cycle"].max()
    stats = {
        "number_of_engines": int(train["unit_id"].nunique()),
        "number_of_rows": int(len(train)),
        "number_of_windows": int((window_metadata["subset"] == "train").sum()),
        "mean_trajectory_length": float(trajectory_lengths.mean()),
        "min_trajectory_length": int(trajectory_lengths.min()),
        "max_trajectory_length": int(trajectory_lengths.max()),
        "average_rul": float(train["rul"].mean()),
        "feature_count": len(feature_names),
    }
    summary_path = output_dir / "baseline_statistics.md"
    lines = ["# Baseline dataset statistics", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in stats.items():
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key.replace('_', ' ').title()} | {rendered} |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sensor_path = output_dir / "sensor_report.csv"
    sensor_report.to_csv(sensor_path, index=False)
    return {"summary": summary_path, "sensor_report": sensor_path}


def generate_engine_profiles(
    train: pd.DataFrame,
    output_dir: Path,
    limit: int | None = None,
    sensors: list[str] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    unit_ids = sorted(train["unit_id"].unique().tolist())
    if limit is not None:
        unit_ids = unit_ids[:limit]
    selected = sensors or ["sensor_11", "sensor_4", "sensor_12"]
    outputs: list[Path] = []
    for unit_id in unit_ids:
        path = output_dir / f"engine_{int(unit_id):03d}.csv"
        columns = ["unit_id", "cycle", "rul", *selected]
        train.loc[train["unit_id"] == unit_id, columns].to_csv(path, index=False)
        outputs.append(path)
    return outputs
