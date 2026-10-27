from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NormalizerArtifact:
    method: str
    feature_names: list[str]
    fit_split: str
    mean: dict[str, float]
    scale: dict[str, float]


def fit_normalizer(
    train_df: pd.DataFrame,
    feature_names: list[str],
    fit_split: str = "train",
) -> tuple[StandardScaler, NormalizerArtifact]:
    fit_frame = (
        train_df[train_df["split"] == fit_split] if "split" in train_df.columns else train_df
    )
    if fit_frame.empty:
        raise ValueError(f"No rows available for normalizer fit split: {fit_split}")
    scaler = StandardScaler()
    scaler.fit(fit_frame[feature_names].to_numpy(dtype=float))
    artifact = NormalizerArtifact(
        method="standard",
        feature_names=feature_names,
        fit_split=fit_split,
        mean=dict(zip(feature_names, scaler.mean_.astype(float), strict=True)),
        scale=dict(zip(feature_names, scaler.scale_.astype(float), strict=True)),
    )
    return scaler, artifact


def transform_features(
    df: pd.DataFrame,
    feature_names: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    frame = df.copy()
    transformed = scaler.transform(frame[feature_names].to_numpy(dtype=float))
    for index, feature_name in enumerate(feature_names):
        frame[feature_name] = transformed[:, index].astype(float)
    return frame


def save_normalizer(
    scaler: StandardScaler,
    artifact: NormalizerArtifact,
    output_dir: Path,
) -> tuple[Path, Path]:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = output_dir / "scaler.pkl"
    artifact_path = output_dir / "normalization.json"
    joblib.dump(scaler, scaler_path)
    artifact_path.write_text(
        json.dumps(asdict(artifact), indent=2, sort_keys=True), encoding="utf-8"
    )
    return scaler_path, artifact_path


def normalizer_to_dict(artifact: NormalizerArtifact) -> dict[str, Any]:
    return asdict(artifact)
