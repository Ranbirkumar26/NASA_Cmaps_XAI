from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from cmapss_xai.constants import DERIVED_TARGET_COLUMNS, ID_COLUMNS, SENSOR_COLUMNS, SETTING_COLUMNS


def build_feature_registry(
    dropped_features: tuple[str, ...] = (),
    selected_features: list[str] | None = None,
) -> dict[str, Any]:
    all_features = [*SETTING_COLUMNS, *SENSOR_COLUMNS]
    dropped = [feature for feature in dropped_features if feature in all_features]
    selected = selected_features or [
        feature for feature in all_features if feature not in set(dropped)
    ]
    return {
        "identifiers": ID_COLUMNS,
        "settings": SETTING_COLUMNS,
        "sensors": SENSOR_COLUMNS,
        "selected_features": selected,
        "dropped": dropped,
        "targets": DERIVED_TARGET_COLUMNS,
        "xai_display_order": selected,
    }


def write_feature_registry(registry: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return output_path


def generate_sensor_report(
    df: pd.DataFrame,
    target_col: str,
    low_variance_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = df[target_col].to_numpy(dtype=float)
    sensor_frame = df[SENSOR_COLUMNS].astype(float)
    mi_values = _mutual_information(sensor_frame, target)
    for sensor in SENSOR_COLUMNS:
        values = sensor_frame[sensor].to_numpy(dtype=float)
        variance = float(np.nanvar(values))
        missing_count = int(np.isnan(values).sum())
        recommendation = (
            "drop" if variance <= low_variance_threshold or missing_count > 0 else "keep"
        )
        rows.append(
            {
                "sensor": sensor,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "variance": variance,
                "missing": missing_count,
                "pearson_correlation": _pearson(values, target),
                "spearman_correlation": _spearman(values, target),
                "mutual_information": mi_values[sensor],
                "recommended_action": recommendation,
            }
        )
    return pd.DataFrame(rows)


def _mutual_information(sensor_frame: pd.DataFrame, target: np.ndarray) -> dict[str, float]:
    variances = sensor_frame.var(axis=0, ddof=0)
    variable_cols = [column for column in sensor_frame.columns if variances[column] > 0]
    scores = dict.fromkeys(sensor_frame.columns, 0.0)
    if not variable_cols:
        return scores
    values = sensor_frame[variable_cols].to_numpy(dtype=float)
    mi = mutual_info_regression(values, target, random_state=42)
    scores.update(dict(zip(variable_cols, mi.astype(float), strict=True)))
    return scores


def _spearman(values: np.ndarray, target: np.ndarray) -> float:
    if _is_constant(values) or _is_constant(target):
        return float("nan")
    correlation = spearmanr(values, target, nan_policy="omit").correlation
    return float(correlation) if correlation is not None else float("nan")


def _pearson(values: np.ndarray, target: np.ndarray) -> float:
    if _is_constant(values) or _is_constant(target):
        return float("nan")
    correlation = np.corrcoef(values, target)[0, 1]
    return float(correlation) if np.isfinite(correlation) else float("nan")


def _is_constant(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return bool(np.allclose(finite, finite[0], rtol=0.0, atol=1e-12))
