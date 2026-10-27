from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from cmapss_xai.config import DatasetConfig, PreprocessingConfig
from cmapss_xai.constants import SUBSET_DESCRIPTIONS
from cmapss_xai.data.download import download_cmapss
from cmapss_xai.data.raw import extract_archive, load_raw_subset
from cmapss_xai.datasets.cmapss import CmapssDataset
from cmapss_xai.preprocessing.features import (
    build_feature_registry,
    generate_sensor_report,
    write_feature_registry,
)
from cmapss_xai.preprocessing.normalization import (
    fit_normalizer,
    normalizer_to_dict,
    save_normalizer,
    transform_features,
)
from cmapss_xai.preprocessing.rul import attach_test_rul, compute_train_rul
from cmapss_xai.preprocessing.splits import assign_unit_splits
from cmapss_xai.preprocessing.windows import make_windows_for_lengths
from cmapss_xai.reports.dataset_card import make_dataset_card
from cmapss_xai.reports.generation import generate_statistics_report
from cmapss_xai.utils.hashing import sha256_files, write_dataset_hash
from cmapss_xai.validation.checks import (
    raise_on_errors,
    validate_processed_tables,
    validate_windows,
)

PREPROCESSING_VERSION = "1.0"


def prepare_subset(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
) -> CmapssDataset:
    _ensure_raw_files(dataset_config)
    raw = load_raw_subset(dataset_config.raw_dir, dataset_config.subset)
    train = compute_train_rul(raw.train, dataset_config.rul_cap)
    train = assign_unit_splits(train, dataset_config.validation_unit_fraction, dataset_config.seed)
    test = attach_test_rul(raw.test, raw.test_rul, dataset_config.rul_cap)
    test["split"] = "test"

    feature_registry = build_feature_registry(preprocessing_config.dropped_features)
    feature_names = list(feature_registry["selected_features"])
    scaler, normalizer = fit_normalizer(train, feature_names, preprocessing_config.fit_split)
    scaled_train = transform_features(train, feature_names, scaler)
    scaled_test = transform_features(test, feature_names, scaler)
    windows = make_windows_for_lengths(
        scaled_train,
        scaled_test,
        feature_names,
        dataset_config.target_column,
        dataset_config.window_candidates,
    )
    issues = validate_processed_tables(train, test)
    issues.extend(validate_windows(windows.metadata, windows.arrays))
    raise_on_errors(issues)

    sensor_report = generate_sensor_report(
        train,
        preprocessing_config.correlation_target,
        preprocessing_config.low_variance_threshold,
    )
    output_dir = dataset_config.processed_dir / dataset_config.subset
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)
    raw.test_rul.to_parquet(output_dir / "test_rul.parquet", index=False)
    windows.metadata.to_parquet(output_dir / "window_metadata.parquet", index=False)
    np.savez_compressed(output_dir / "windows.npz", **windows.arrays)
    sensor_report.to_csv(output_dir / "sensor_report.csv", index=False)
    sensor_report.to_parquet(output_dir / "sensor_report.parquet", index=False)
    write_feature_registry(feature_registry, output_dir / "feature_registry.yaml")
    save_normalizer(scaler, normalizer, output_dir)

    metadata = _build_metadata(
        dataset_config,
        normalizer_to_dict(normalizer),
        feature_names,
        train,
        test,
        output_dir,
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["dataset_hash"] = fingerprint_subset(dataset_config)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    card = make_dataset_card(metadata, dataset_config.reports_dir / "dataset_card")
    shutil.copyfile(card, output_dir / "dataset_card.md")
    generate_statistics_report(
        train,
        sensor_report,
        windows.metadata,
        feature_names,
        dataset_config.reports_dir / "statistics",
    )
    (output_dir / "validation_report.txt").write_text("OK\n", encoding="utf-8")
    return CmapssDataset(
        subset=dataset_config.subset,
        raw_train=raw.train,
        raw_test=raw.test,
        processed_train=train,
        processed_test=test,
        test_rul=raw.test_rul,
        metadata=metadata,
        scaler=scaler,
        feature_registry=feature_registry,
        window_arrays=windows.arrays,
        window_metadata=windows.metadata,
        feature_names=feature_names,
        sensor_statistics=sensor_report,
    )


def load_prepared_subset(dataset_config: DatasetConfig) -> CmapssDataset:
    output_dir = dataset_config.processed_dir / dataset_config.subset
    train = pd.read_parquet(output_dir / "train.parquet")
    test = pd.read_parquet(output_dir / "test.parquet")
    test_rul = pd.read_parquet(output_dir / "test_rul.parquet")
    window_metadata = pd.read_parquet(output_dir / "window_metadata.parquet")
    arrays = dict(np.load(output_dir / "windows.npz"))
    sensor_report = pd.read_parquet(output_dir / "sensor_report.parquet")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    scaler = joblib.load(output_dir / "scaler.pkl")
    feature_registry = yaml.safe_load(
        (output_dir / "feature_registry.yaml").read_text(encoding="utf-8")
    )
    return CmapssDataset(
        subset=dataset_config.subset,
        raw_train=train,
        raw_test=test,
        processed_train=train,
        processed_test=test,
        test_rul=test_rul,
        metadata=metadata,
        scaler=scaler,
        feature_registry=feature_registry,
        window_arrays=arrays,
        window_metadata=window_metadata,
        feature_names=list(feature_registry["selected_features"]),
        sensor_statistics=sensor_report,
    )


def validate_subset(dataset_config: DatasetConfig) -> list[str]:
    dataset = load_prepared_subset(dataset_config)
    issues = validate_processed_tables(dataset.processed_train, dataset.processed_test)
    issues.extend(validate_windows(dataset.window_metadata, dataset.window_arrays))
    output_dir = dataset_config.processed_dir / dataset_config.subset
    lines = [f"{issue.severity.upper()} {issue.code}: {issue.message}" for issue in issues] or [
        "OK"
    ]
    (output_dir / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    raise_on_errors(issues)
    return lines


def fingerprint_subset(dataset_config: DatasetConfig) -> str:
    output_dir = dataset_config.processed_dir / dataset_config.subset
    paths = [
        output_dir / "train.parquet",
        output_dir / "test.parquet",
        output_dir / "test_rul.parquet",
        output_dir / "window_metadata.parquet",
        output_dir / "feature_registry.yaml",
        Path("configs/dataset.yaml"),
        Path("configs/preprocessing.yaml"),
    ]
    resolved = [path if path.is_absolute() else Path.cwd() / path for path in paths]
    return write_dataset_hash(resolved, output_dir / "dataset_hash.txt")


def _ensure_raw_files(dataset_config: DatasetConfig) -> None:
    train_path = dataset_config.raw_dir / "CMAPSSData" / f"train_{dataset_config.subset}.txt"
    test_path = dataset_config.raw_dir / "CMAPSSData" / f"test_{dataset_config.subset}.txt"
    if train_path.exists() and test_path.exists():
        return
    zip_path = download_cmapss(dataset_config)
    extract_archive(zip_path, dataset_config.raw_dir)


def _build_metadata(
    dataset_config: DatasetConfig,
    normalizer: dict[str, Any],
    feature_names: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    output_hashes = {
        name: sha256_files([output_dir / name])
        for name in [
            "train.parquet",
            "test.parquet",
            "test_rul.parquet",
            "window_metadata.parquet",
            "windows.npz",
            "feature_registry.yaml",
            "normalization.json",
        ]
        if (output_dir / name).exists()
    }
    return {
        "dataset_name": "NASA C-MAPSS Jet Engine Simulated Data",
        "dataset_version": dataset_config.dataset_version,
        "preprocessing_version": PREPROCESSING_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "source_url": dataset_config.source_url,
        "source_page": dataset_config.source_page,
        "citation": dataset_config.citation,
        "license_note": dataset_config.license_note,
        "subset": dataset_config.subset,
        "subset_description": SUBSET_DESCRIPTIONS.get(dataset_config.subset, {}),
        "rul_cap": dataset_config.rul_cap,
        "window_default": dataset_config.window_default,
        "window_lengths": list(dataset_config.window_candidates),
        "split_seed": dataset_config.seed,
        "validation_unit_fraction": dataset_config.validation_unit_fraction,
        "feature_names": feature_names,
        "row_counts": {"train": int(len(train)), "test": int(len(test))},
        "unit_counts": {
            "train": int(train["unit_id"].nunique()),
            "test": int(test["unit_id"].nunique()),
        },
        "normalization": normalizer,
        "output_hashes": output_hashes,
    }


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unavailable"
