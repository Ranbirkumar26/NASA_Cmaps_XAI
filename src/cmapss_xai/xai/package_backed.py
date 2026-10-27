from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PackageBackedXaiConfig:
    background_size: int = 64
    shap_samples: int = 4
    lime_samples: int = 8
    dice_samples: int = 2
    random_state: int = 42


def shap_sensor_attribution(
    model: Any,
    background: np.ndarray,
    samples: np.ndarray,
    feature_names: list[str],
    config: PackageBackedXaiConfig,
) -> np.ndarray:
    shap = _require_shap()
    background_flat = _flatten(background[: config.background_size]).astype(np.float64)
    samples_used = samples[: config.shap_samples]
    sample_flat = _flatten(samples_used)
    masker = shap.maskers.Independent(background_flat)
    explainer = shap.Explainer(
        model.predict,
        masker,
        algorithm="permutation",
        seed=config.random_state,
    )
    max_evals = max(2 * sample_flat.shape[1] + 1, 1500)
    values = explainer(sample_flat, max_evals=max_evals).values
    return _aggregate_flat_attribution(
        np.asarray(values, dtype=float),
        samples.shape[1],
        feature_names,
    )


def lime_sensor_attribution(
    model: Any,
    background: np.ndarray,
    samples: np.ndarray,
    feature_names: list[str],
    config: PackageBackedXaiConfig,
) -> np.ndarray:
    lime_tabular = _require_lime()
    background_flat = _flatten(background[: config.background_size])
    samples_used = samples[: config.lime_samples]
    flat_names = flat_feature_names(samples.shape[1], feature_names)
    explainer = lime_tabular.LimeTabularExplainer(
        background_flat,
        feature_names=flat_names,
        mode="regression",
        discretize_continuous=False,
        random_state=config.random_state,
    )
    rows = []
    for sample in _flatten(samples_used):
        explanation = explainer.explain_instance(
            sample,
            model.predict,
            num_features=len(flat_names),
        )
        values = np.zeros(len(flat_names), dtype=float)
        local_values = next(iter(explanation.local_exp.values()))
        for feature_idx, weight in local_values:
            values[int(feature_idx)] = float(weight)
        rows.append(values)
    return _aggregate_flat_attribution(
        np.asarray(rows, dtype=float),
        samples.shape[1],
        feature_names,
    )


def dice_counterfactuals(
    model: Any,
    background: np.ndarray,
    y_background: np.ndarray,
    samples: np.ndarray,
    sample_indices: np.ndarray,
    feature_names: list[str],
    config: PackageBackedXaiConfig,
    min_gain: float = 10.0,
) -> pd.DataFrame:
    dice_ml = _require_dice()
    window_length = samples.shape[1]
    flat_names = flat_feature_names(window_length, feature_names)
    background_flat = _flatten(background[: config.background_size])
    y_values = np.asarray(y_background[: config.background_size], dtype=float)
    train_frame = pd.DataFrame(background_flat, columns=flat_names)
    train_frame["rul"] = y_values
    data_interface = dice_ml.Data(
        dataframe=train_frame,
        continuous_features=flat_names,
        outcome_name="rul",
    )
    model_interface = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")
    explainer = dice_ml.Dice(data_interface, model_interface, method="random")
    query = pd.DataFrame(
        _flatten(samples[: config.dice_samples]).astype(np.float64),
        columns=flat_names,
    )
    predictions = np.asarray(model.predict(query.to_numpy()), dtype=float)
    rows: list[dict[str, Any]] = []
    for row_idx, prediction in enumerate(predictions):
        desired_range = [
            float(prediction + min_gain),
            float(max(prediction + min_gain + 1.0, 125.0)),
        ]
        row = {
            "sample_index": int(sample_indices[row_idx]),
            "current_prediction": float(prediction),
            "counterfactual_prediction": np.nan,
            "expected_rul_gain": np.nan,
            "changed_sensors": "",
            "method": "dice_ml_random",
            "status": "not_found",
        }
        try:
            generated = explainer.generate_counterfactuals(
                query.iloc[[row_idx]],
                total_CFs=1,
                desired_range=desired_range,
                verbose=False,
            )
            final = generated.cf_examples_list[0].final_cfs_df
            if final is not None and not final.empty:
                cf_features = final.iloc[0][flat_names].to_numpy(dtype=float)
                cf_prediction = float(model.predict(cf_features.reshape(1, -1))[0])
                row.update(
                    {
                        "counterfactual_prediction": cf_prediction,
                        "expected_rul_gain": cf_prediction - float(prediction),
                        "changed_sensors": ", ".join(
                            _changed_sensor_names(
                                query.iloc[row_idx].to_numpy(dtype=float),
                                cf_features,
                                window_length,
                                feature_names,
                            )
                        ),
                        "status": "found",
                    }
                )
        except (ValueError, RuntimeError, TypeError, IndexError, KeyError) as exc:
            row["status"] = f"failed: {exc.__class__.__name__}"
        rows.append(row)
    return pd.DataFrame(rows)


def flat_feature_names(window_length: int, feature_names: list[str]) -> list[str]:
    return [
        f"t{time_idx:02d}_{feature_name}"
        for time_idx in range(window_length)
        for feature_name in feature_names
    ]


def _aggregate_flat_attribution(
    values: np.ndarray,
    window_length: int,
    feature_names: list[str],
) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D attribution values, got shape {values.shape}")
    reshaped = values.reshape(values.shape[0], window_length, len(feature_names))
    return np.sum(reshaped, axis=1)


def _changed_sensor_names(
    current: np.ndarray,
    counterfactual: np.ndarray,
    window_length: int,
    feature_names: list[str],
    top_k: int = 5,
) -> list[str]:
    delta = np.abs(counterfactual - current).reshape(window_length, len(feature_names))
    scores = np.mean(delta, axis=0)
    top = np.argsort(scores)[-top_k:][::-1]
    return [feature_names[int(idx)] for idx in top if scores[int(idx)] > 0]


def _flatten(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(x.shape[0], -1)


def _require_shap() -> Any:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "Package-backed SHAP requires installing the deep-xai extra: "
            'python3.11 -m pip install -e ".[deep-xai]"'
        ) from exc
    return shap


def _require_lime() -> Any:
    try:
        from lime import lime_tabular
    except ImportError as exc:
        raise RuntimeError(
            "Package-backed LIME requires installing the deep-xai extra: "
            'python3.11 -m pip install -e ".[deep-xai]"'
        ) from exc
    return lime_tabular


def _require_dice() -> Any:
    try:
        import dice_ml
    except ImportError as exc:
        raise RuntimeError(
            "Package-backed DiCE requires installing the deep-xai extra: "
            'python3.11 -m pip install -e ".[deep-xai]"'
        ) from exc
    return dice_ml
