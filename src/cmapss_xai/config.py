from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cmapss_xai.paths import project_path


@dataclass(frozen=True)
class DatasetConfig:
    subset: str
    dataset_version: str
    source_url: str
    source_page: str
    citation: str
    license_note: str
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    results_dir: Path
    validation_unit_fraction: float
    seed: int
    rul_cap: int
    target_column: str
    window_default: int
    window_candidates: tuple[int, ...]


@dataclass(frozen=True)
class PreprocessingConfig:
    normalization_method: str
    fit_split: str
    low_variance_threshold: float
    dropped_features: tuple[str, ...]
    auto_drop_low_variance: bool
    correlation_target: str


def load_yaml(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_dataset_config(path: str | Path = "configs/dataset.yaml") -> DatasetConfig:
    raw = load_yaml(path)
    dataset = raw["dataset"]
    paths = raw["paths"]
    splits = raw["splits"]
    targets = raw["targets"]
    windows = raw["windows"]
    return DatasetConfig(
        subset=str(dataset["subset"]).upper(),
        dataset_version=str(dataset["dataset_version"]),
        source_url=str(dataset["source_url"]),
        source_page=str(dataset["source_page"]),
        citation=str(dataset["citation"]),
        license_note=str(dataset["license_note"]),
        raw_dir=project_path(paths["raw_dir"]),
        processed_dir=project_path(paths["processed_dir"]),
        reports_dir=project_path(paths["reports_dir"]),
        results_dir=project_path(paths["results_dir"]),
        validation_unit_fraction=float(splits["validation_unit_fraction"]),
        seed=int(splits["seed"]),
        rul_cap=int(targets["rul_cap"]),
        target_column=str(targets["target_column"]),
        window_default=int(windows["default"]),
        window_candidates=tuple(int(value) for value in windows["candidates"]),
    )


def load_preprocessing_config(
    path: str | Path = "configs/preprocessing.yaml",
) -> PreprocessingConfig:
    raw = load_yaml(path)
    normalization = raw["normalization"]
    features = raw["features"]
    return PreprocessingConfig(
        normalization_method=str(normalization["method"]),
        fit_split=str(normalization["fit_split"]),
        low_variance_threshold=float(features["low_variance_threshold"]),
        dropped_features=tuple(str(value) for value in features["dropped_features"]),
        auto_drop_low_variance=bool(features["auto_drop_low_variance"]),
        correlation_target=str(features["correlation_target"]),
    )


def override_subset(config: DatasetConfig, subset: str | None) -> DatasetConfig:
    if subset is None:
        return config
    return DatasetConfig(**{**config.__dict__, "subset": subset.upper()})
