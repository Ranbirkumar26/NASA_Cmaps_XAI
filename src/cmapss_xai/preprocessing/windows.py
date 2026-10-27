from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowedDataset:
    arrays: dict[str, np.ndarray]
    metadata: pd.DataFrame


def make_windows_for_lengths(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    window_lengths: tuple[int, ...],
) -> WindowedDataset:
    arrays: dict[str, np.ndarray] = {}
    metadata_frames: list[pd.DataFrame] = []
    for length in window_lengths:
        for name, frame in (("train", train_df), ("test", test_df)):
            x, y, meta = make_windows(frame, feature_names, target_col, length, name)
            arrays[f"X_{name}_{length}"] = x
            arrays[f"y_{name}_{length}"] = y
            metadata_frames.append(meta)
    metadata = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    if not metadata.empty:
        metadata.insert(0, "window_id", range(1, len(metadata) + 1))
    return WindowedDataset(arrays=arrays, metadata=metadata)


def make_windows(
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    window_length: int,
    subset: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, object]] = []
    for unit_id, unit_df in df.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=True):
        unit_features = unit_df[feature_names].to_numpy(dtype=np.float32)
        unit_targets = unit_df[target_col].to_numpy(dtype=np.float32)
        cycles = unit_df["cycle"].to_numpy(dtype=np.int64)
        split = str(unit_df["split"].iloc[0]) if "split" in unit_df.columns else subset
        if len(unit_df) < window_length:
            continue
        for start in range(0, len(unit_df) - window_length + 1):
            end = start + window_length
            target_index = end - 1
            windows.append(unit_features[start:end])
            targets.append(float(unit_targets[target_index]))
            metadata.append(
                {
                    "array_key": f"X_{subset}_{window_length}",
                    "target_key": f"y_{subset}_{window_length}",
                    "array_index": len(windows) - 1,
                    "subset": subset,
                    "split": split,
                    "unit_id": int(unit_id),
                    "start_cycle": int(cycles[start]),
                    "end_cycle": int(cycles[target_index]),
                    "target_rul": float(unit_targets[target_index]),
                    "window_length": int(window_length),
                }
            )
    if not windows:
        return (
            np.empty((0, window_length, len(feature_names)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DataFrame(metadata),
        )
    return (
        np.stack(windows).astype(np.float32),
        np.array(targets, dtype=np.float32),
        pd.DataFrame(metadata),
    )
