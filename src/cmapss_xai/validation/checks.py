from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cmapss_xai.constants import RAW_COLUMNS


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


def validate_processed_tables(train: pd.DataFrame, test: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, frame in {"train": train, "test": test}.items():
        issues.extend(_validate_schema(name, frame))
        issues.extend(_validate_unit_cycles(name, frame))
        issues.extend(_validate_numeric(name, frame))
    return issues


def validate_windows(
    window_metadata: pd.DataFrame, arrays: dict[str, np.ndarray]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required = {
        "array_key",
        "array_index",
        "unit_id",
        "start_cycle",
        "end_cycle",
        "window_length",
        "target_rul",
    }
    missing = required - set(window_metadata.columns)
    if missing:
        return [
            ValidationIssue(
                "error", "window_metadata_schema", f"Missing columns: {sorted(missing)}"
            )
        ]
    spans = window_metadata["end_cycle"] - window_metadata["start_cycle"] + 1
    if (spans != window_metadata["window_length"]).any():
        issues.append(ValidationIssue("error", "window_cycle_span", "Window spans mismatch"))
    for array_key, group in window_metadata.groupby("array_key"):
        if array_key not in arrays:
            issues.append(
                ValidationIssue("error", "window_array_missing", f"Missing array {array_key}")
            )
            continue
        if int(group["array_index"].max()) >= len(arrays[array_key]):
            issues.append(
                ValidationIssue("error", "window_array_index", f"Bad index for {array_key}")
            )
    return issues


def raise_on_errors(issues: list[ValidationIssue]) -> None:
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise ValueError("\n".join(f"{issue.code}: {issue.message}" for issue in errors))


def _validate_schema(name: str, frame: pd.DataFrame) -> list[ValidationIssue]:
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        return [ValidationIssue("error", f"{name}_schema", f"Missing columns: {missing}")]
    return []


def _validate_unit_cycles(name: str, frame: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    duplicated = frame.duplicated(["unit_id", "cycle"]).sum()
    if duplicated:
        issues.append(
            ValidationIssue("error", f"{name}_duplicated_unit_cycle", f"{duplicated} rows")
        )
    if (frame["cycle"] < 1).any():
        issues.append(ValidationIssue("error", f"{name}_negative_cycle", "Cycle values below 1"))
    first_cycles = frame.groupby("unit_id")["cycle"].min()
    bad_starts = first_cycles[first_cycles != 1]
    if not bad_starts.empty:
        issues.append(
            ValidationIssue("error", f"{name}_cycle_start", f"Units: {bad_starts.index.tolist()}")
        )
    for unit_id, unit_df in frame.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=True):
        cycles = unit_df["cycle"].to_numpy()
        if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
            issues.append(ValidationIssue("error", f"{name}_cycle_monotonic", f"Unit: {unit_id}"))
            break
    return issues


def _validate_numeric(name: str, frame: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.isna().any().any():
        issues.append(ValidationIssue("error", f"{name}_nan", "NaN values found"))
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        issues.append(ValidationIssue("error", f"{name}_inf", "Infinite values found"))
    bad_dtypes = [
        column for column in RAW_COLUMNS if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if bad_dtypes:
        issues.append(ValidationIssue("error", f"{name}_dtypes", f"Invalid dtypes: {bad_dtypes}"))
    return issues
