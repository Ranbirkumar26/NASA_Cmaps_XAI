from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPRegressor

from cmapss_xai.config import DatasetConfig, PreprocessingConfig
from cmapss_xai.evaluation.metrics import mae, nasa_score, r2, rmse
from cmapss_xai.models.sequence import TorchSequenceRegressor, TorchTrainingConfig
from cmapss_xai.pipeline import load_prepared_subset, prepare_subset
from cmapss_xai.xai.integrated_gradients import integrated_gradients_attribution
from cmapss_xai.xai.package_backed import (
    PackageBackedXaiConfig,
    dice_counterfactuals,
    lime_sensor_attribution,
    shap_sensor_attribution,
)


@dataclass(frozen=True)
class XaiRunConfig:
    subset: str = "FD001"
    window_length: int = 30
    max_train_windows: int = 6000
    max_eval_windows: int = 3000
    xai_samples: int = 40
    random_state: int = 42
    include_deep_models: bool = False
    deep_epochs: int = 4
    deep_batch_size: int = 128
    deep_hidden_size: int = 48
    deep_learning_rate: float = 1e-3
    deep_dropout: float = 0.1
    exact_xai: bool = False
    exact_xai_background: int = 64
    exact_shap_samples: int = 4
    exact_lime_samples: int = 8
    exact_dice_samples: int = 2
    integrated_gradients: bool = False
    benchmark_explainers: bool = False
    prediction_intervals: bool = False
    tuning_grid: bool = False


@dataclass(frozen=True)
class WindowBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    meta_train: pd.DataFrame
    x_validation: np.ndarray
    y_validation: np.ndarray
    meta_validation: pd.DataFrame
    x_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame
    feature_names: list[str]
    baseline_window: np.ndarray
    low_variance_sensors: list[str]


def run_xai_project(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    run_config: XaiRunConfig,
) -> dict[str, Path]:
    dataset = _load_or_prepare(dataset_config, preprocessing_config)
    bundle = build_window_bundle(dataset, run_config)
    output_dir = dataset_config.results_dir / "xai" / dataset_config.subset
    report_dir = dataset_config.reports_dir / "xai_evaluation" / dataset_config.subset
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    models, model_metrics = train_benchmark_models(bundle, output_dir, run_config)
    best_name = str(model_metrics.sort_values("validation_rmse").iloc[0]["model"])
    best_model = models[best_name]
    explanations = generate_explanations(best_model, bundle, output_dir, run_config)
    quality = evaluate_explanation_quality(best_model, bundle, explanations, output_dir)
    trust = evaluate_trustworthiness(models, best_name, bundle, explanations, output_dir)
    advisor = generate_decision_support(best_model, bundle, explanations, output_dir)
    prediction_interval_path = generate_prediction_intervals(models, bundle, output_dir)
    subgroup_path = write_subgroup_metrics(models[best_name], bundle, output_dir)
    fairness_mitigation_path = write_balanced_sampling_comparison(bundle, output_dir, run_config)
    if run_config.tuning_grid:
        run_tuning_grid(bundle, output_dir, run_config)
    framework = build_dimension_framework(model_metrics, quality, trust, advisor, output_dir)
    report_path = write_xai_report(
        report_dir,
        dataset_config,
        run_config,
        model_metrics,
        best_name,
        quality,
        trust,
        framework,
    )
    result = {
        "model_metrics": output_dir / "model_benchmark.csv",
        "quality": output_dir / "xai_quality.json",
        "trust": output_dir / "trustworthiness.json",
        "framework": output_dir / "xai_evaluation_framework.csv",
        "advisor": output_dir / "maintenance_advisor.csv",
        "prediction_intervals": prediction_interval_path,
        "subgroup_metrics": subgroup_path,
        "fairness_mitigation": fairness_mitigation_path,
        "report": report_path,
    }
    if run_config.benchmark_explainers:
        result["explainer_runtime"] = output_dir / "explainer_runtime.csv"
    if run_config.integrated_gradients:
        ig_dir = output_dir / "explanations" / "integrated_gradients"
        result["integrated_gradients_sensor"] = (
            ig_dir / "integrated_gradients_sensor_importance.csv"
        )
        result["integrated_gradients_temporal"] = (
            ig_dir / "integrated_gradients_temporal_importance.csv"
        )
        result["integrated_gradients_local"] = ig_dir / "integrated_gradients_local_examples.csv"
    if run_config.tuning_grid:
        result["tuning_grid"] = output_dir / "tuning_grid.csv"
    if run_config.exact_xai:
        outputs = output_dir / "explanations" / "package_backed"
        result["package_shap"] = outputs / "shap_sensor_importance.csv"
        result["package_lime"] = outputs / "lime_sensor_importance.csv"
        result["package_dice"] = outputs / "dice_counterfactuals.csv"
    return result


def build_window_bundle(dataset: Any, run_config: XaiRunConfig) -> WindowBundle:
    length = run_config.window_length
    x_train_all = dataset.window_arrays[f"X_train_{length}"]
    y_train_all = dataset.window_arrays[f"y_train_{length}"]
    train_meta_all = dataset.window_metadata[
        dataset.window_metadata["array_key"] == f"X_train_{length}"
    ]
    test_meta = dataset.window_metadata[
        dataset.window_metadata["array_key"] == f"X_test_{length}"
    ].copy()
    train_mask = train_meta_all["split"].to_numpy() == "train"
    validation_mask = train_meta_all["split"].to_numpy() == "validation"

    x_train = x_train_all[train_mask]
    y_train = y_train_all[train_mask]
    meta_train = train_meta_all.loc[train_mask].reset_index(drop=True)
    x_validation = x_train_all[validation_mask]
    y_validation = y_train_all[validation_mask]
    meta_validation = train_meta_all.loc[validation_mask].reset_index(drop=True)
    x_test = dataset.window_arrays[f"X_test_{length}"]
    y_test = dataset.window_arrays[f"y_test_{length}"]
    low_variance = dataset.sensor_statistics.loc[
        dataset.sensor_statistics["recommended_action"] == "drop", "sensor"
    ].tolist()
    return WindowBundle(
        x_train=x_train,
        y_train=y_train,
        meta_train=meta_train,
        x_validation=x_validation,
        y_validation=y_validation,
        meta_validation=meta_validation,
        x_test=x_test,
        y_test=y_test,
        meta_test=test_meta.reset_index(drop=True),
        feature_names=list(dataset.feature_names),
        baseline_window=np.mean(x_train, axis=0),
        low_variance_sensors=low_variance,
    )


def train_benchmark_models(
    bundle: WindowBundle,
    output_dir: Path,
    run_config: XaiRunConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(run_config.random_state)
    x_train = _flatten(bundle.x_train)
    y_train = bundle.y_train
    if len(x_train) > run_config.max_train_windows:
        indices = rng.choice(len(x_train), size=run_config.max_train_windows, replace=False)
        x_fit = x_train[indices]
        x_fit_sequence = bundle.x_train[indices]
        y_fit = y_train[indices]
    else:
        x_fit = x_train
        x_fit_sequence = bundle.x_train
        y_fit = y_train
    models: dict[str, Any] = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=70,
            max_depth=14,
            min_samples_leaf=3,
            random_state=run_config.random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=run_config.random_state,
        ),
        "mlp_neural_regressor": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=60,
            early_stopping=True,
            random_state=run_config.random_state,
        ),
    }
    if run_config.include_deep_models:
        torch_config = TorchTrainingConfig(
            epochs=run_config.deep_epochs,
            batch_size=run_config.deep_batch_size,
            learning_rate=run_config.deep_learning_rate,
            hidden_size=run_config.deep_hidden_size,
            dropout=run_config.deep_dropout,
            random_state=run_config.random_state,
        )
        models.update(
            {
                "lstm_sequence": TorchSequenceRegressor(
                    model_type="lstm",
                    window_shape=bundle.x_train.shape[1:],
                    config=torch_config,
                ),
                "gru_sequence": TorchSequenceRegressor(
                    model_type="gru",
                    window_shape=bundle.x_train.shape[1:],
                    config=torch_config,
                ),
                "transformer_sequence": TorchSequenceRegressor(
                    model_type="transformer",
                    window_shape=bundle.x_train.shape[1:],
                    config=torch_config,
                ),
            }
        )
    rows: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if _expects_sequence_input(model):
            model.fit(x_fit_sequence, y_fit)
            model.save(model_dir / f"{name}.pt")
        else:
            model.fit(x_fit, y_fit)
            joblib.dump(model, model_dir / f"{name}.joblib")
        validation_pred = _predict_window_model(model, bundle.x_validation)
        test_pred = _predict_window_model(model, bundle.x_test)
        predictions[name] = test_pred
        rows.append(
            {
                "model": name,
                "train_windows": len(y_fit),
                "validation_rmse": rmse(bundle.y_validation, validation_pred),
                "validation_mae": mae(bundle.y_validation, validation_pred),
                "validation_r2": r2(bundle.y_validation, validation_pred),
                "validation_nasa_score": nasa_score(bundle.y_validation, validation_pred),
                "test_rmse": rmse(bundle.y_test, test_pred),
                "test_mae": mae(bundle.y_test, test_pred),
                "test_r2": r2(bundle.y_test, test_pred),
                "test_nasa_score": nasa_score(bundle.y_test, test_pred),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("validation_rmse").reset_index(drop=True)
    metrics.to_csv(output_dir / "model_benchmark.csv", index=False)
    prediction_frame = bundle.meta_test[
        ["unit_id", "start_cycle", "end_cycle", "target_rul"]
    ].copy()
    for name, pred in predictions.items():
        prediction_frame[f"pred_{name}"] = pred
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    return models, metrics


def generate_explanations(
    model: Any,
    bundle: WindowBundle,
    output_dir: Path,
    run_config: XaiRunConfig,
) -> dict[str, Any]:
    sample_indices = _select_xai_samples(bundle.y_test, run_config.xai_samples)
    samples = bundle.x_test[sample_indices]
    meta = bundle.meta_test.iloc[sample_indices].reset_index(drop=True)
    runtime_rows: list[dict[str, Any]] = []
    sensor_attr = _timed(
        "sensor_occlusion",
        lambda: sensor_occlusion_attribution(
            model, samples, bundle.feature_names, bundle.baseline_window
        ),
        runtime_rows,
        run_config.benchmark_explainers,
        len(samples),
    )
    time_attr = _timed(
        "temporal_occlusion",
        lambda: temporal_occlusion_attribution(model, samples, bundle.baseline_window),
        runtime_rows,
        run_config.benchmark_explainers,
        len(samples),
    )
    lime_attr = _timed(
        "lime_style_surrogate",
        lambda: lime_style_attribution(
            model,
            samples,
            bundle.feature_names,
            bundle.baseline_window,
            random_state=run_config.random_state,
        ),
        runtime_rows,
        run_config.benchmark_explainers,
        len(samples),
    )
    global_sensor = _global_importance_frame(
        sensor_attr, bundle.feature_names, "shap_style_occlusion"
    )
    global_lime = _global_importance_frame(lime_attr, bundle.feature_names, "lime_style_surrogate")
    global_time = pd.DataFrame(
        {
            "time_index": np.arange(samples.shape[1]),
            "mean_abs_attribution": np.mean(np.abs(time_attr), axis=0),
            "method": "temporal_occlusion",
        }
    ).sort_values("mean_abs_attribution", ascending=False)
    local_rows = []
    for row_idx, sample_index in enumerate(sample_indices):
        top_sensors = _top_names(sensor_attr[row_idx], bundle.feature_names, top_k=5)
        local_rows.append(
            {
                "sample_index": int(sample_index),
                "unit_id": int(meta.loc[row_idx, "unit_id"]),
                "start_cycle": int(meta.loc[row_idx, "start_cycle"]),
                "end_cycle": int(meta.loc[row_idx, "end_cycle"]),
                "target_rul": float(meta.loc[row_idx, "target_rul"]),
                "prediction": float(
                    _predict_window_model(model, samples[row_idx : row_idx + 1])[0]
                ),
                "top_sensors": ", ".join(top_sensors),
                "top_time_index": int(np.argmax(np.abs(time_attr[row_idx]))),
            }
    )
    local_examples = pd.DataFrame(local_rows)
    counterfactuals = nearest_counterfactuals(model, samples, bundle, sample_indices)
    package_backed = None
    if run_config.exact_xai:
        package_backed = generate_package_backed_explanations(
            model,
            bundle,
            samples,
            sample_indices,
            output_dir,
            run_config,
            runtime_rows,
        )
    integrated_gradients = None
    if run_config.integrated_gradients:
        ig_meta = meta.copy()
        ig_meta["sample_index"] = sample_indices
        if _expects_sequence_input(model):
            integrated_gradients = _timed(
                "integrated_gradients",
                lambda: generate_integrated_gradients_explanations(
                    model,
                    bundle,
                    samples,
                    ig_meta,
                    output_dir,
                ),
                runtime_rows,
                run_config.benchmark_explainers,
                len(samples),
            )
        else:
            integrated_gradients = write_integrated_gradients_skipped(output_dir, str(type(model)))

    explanations_dir = output_dir / "explanations"
    explanations_dir.mkdir(parents=True, exist_ok=True)
    global_sensor.to_csv(explanations_dir / "global_sensor_importance.csv", index=False)
    global_lime.to_csv(explanations_dir / "global_lime_importance.csv", index=False)
    global_time.to_csv(explanations_dir / "global_temporal_importance.csv", index=False)
    local_examples.to_csv(explanations_dir / "local_examples.csv", index=False)
    counterfactuals.to_csv(explanations_dir / "counterfactuals.csv", index=False)
    if run_config.benchmark_explainers:
        pd.DataFrame(runtime_rows).to_csv(output_dir / "explainer_runtime.csv", index=False)
    return {
        "sample_indices": sample_indices,
        "samples": samples,
        "meta": meta,
        "sensor_attr": sensor_attr,
        "time_attr": time_attr,
        "lime_attr": lime_attr,
        "global_sensor": global_sensor,
        "global_lime": global_lime,
        "global_time": global_time,
        "local_examples": local_examples,
        "counterfactuals": counterfactuals,
        "package_backed": package_backed,
        "integrated_gradients": integrated_gradients,
    }


def generate_package_backed_explanations(
    model: Any,
    bundle: WindowBundle,
    samples: np.ndarray,
    sample_indices: np.ndarray,
    output_dir: Path,
    run_config: XaiRunConfig,
    runtime_rows: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    config = PackageBackedXaiConfig(
        background_size=run_config.exact_xai_background,
        shap_samples=run_config.exact_shap_samples,
        lime_samples=run_config.exact_lime_samples,
        dice_samples=run_config.exact_dice_samples,
        random_state=run_config.random_state,
    )
    package_dir = output_dir / "explanations" / "package_backed"
    package_dir.mkdir(parents=True, exist_ok=True)
    shap_attr = _timed(
        "shap_package",
        lambda: shap_sensor_attribution(
            model,
            bundle.x_train,
            samples,
            bundle.feature_names,
            config,
        ),
        runtime_rows,
        run_config.benchmark_explainers,
        config.shap_samples,
    )
    lime_attr = _timed(
        "lime_package",
        lambda: lime_sensor_attribution(
            model,
            bundle.x_train,
            samples,
            bundle.feature_names,
            config,
        ),
        runtime_rows,
        run_config.benchmark_explainers,
        config.lime_samples,
    )
    dice_rows = _timed(
        "dice_ml_random",
        lambda: dice_counterfactuals(
            model,
            bundle.x_train,
            bundle.y_train,
            samples,
            sample_indices,
            bundle.feature_names,
            config,
        ),
        runtime_rows,
        run_config.benchmark_explainers,
        config.dice_samples,
    )
    shap_frame = _global_importance_frame(shap_attr, bundle.feature_names, "shap_package")
    lime_frame = _global_importance_frame(lime_attr, bundle.feature_names, "lime_package")
    shap_frame.to_csv(package_dir / "shap_sensor_importance.csv", index=False)
    lime_frame.to_csv(package_dir / "lime_sensor_importance.csv", index=False)
    dice_rows.to_csv(package_dir / "dice_counterfactuals.csv", index=False)
    return {
        "shap": shap_frame,
        "lime": lime_frame,
        "dice": dice_rows,
    }


def generate_integrated_gradients_explanations(
    model: Any,
    bundle: WindowBundle,
    samples: np.ndarray,
    meta: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    ig_dir = output_dir / "explanations" / "integrated_gradients"
    ig_dir.mkdir(parents=True, exist_ok=True)
    result = integrated_gradients_attribution(
        model,
        samples,
        bundle.baseline_window,
        bundle.feature_names,
        meta,
    )
    sensor_frame = _global_importance_frame(
        result.sensor_attribution,
        bundle.feature_names,
        "integrated_gradients",
    )
    temporal_frame = pd.DataFrame(
        {
            "time_index": np.arange(result.temporal_attribution.shape[1]),
            "mean_abs_attribution": np.mean(np.abs(result.temporal_attribution), axis=0),
            "method": "integrated_gradients",
        }
    ).sort_values("mean_abs_attribution", ascending=False)
    sensor_frame.to_csv(ig_dir / "integrated_gradients_sensor_importance.csv", index=False)
    temporal_frame.to_csv(ig_dir / "integrated_gradients_temporal_importance.csv", index=False)
    result.local_examples.to_csv(ig_dir / "integrated_gradients_local_examples.csv", index=False)
    return {
        "sensor": sensor_frame,
        "temporal": temporal_frame,
        "local": result.local_examples,
    }


def write_integrated_gradients_skipped(
    output_dir: Path,
    model_name: str,
) -> dict[str, pd.DataFrame]:
    ig_dir = output_dir / "explanations" / "integrated_gradients"
    ig_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{"status": "skipped", "reason": "best_model_is_not_torch_sequence", "model": model_name}]
    )
    frame.to_csv(ig_dir / "integrated_gradients_sensor_importance.csv", index=False)
    frame.to_csv(ig_dir / "integrated_gradients_temporal_importance.csv", index=False)
    frame.to_csv(ig_dir / "integrated_gradients_local_examples.csv", index=False)
    return {"sensor": frame, "temporal": frame, "local": frame}


def sensor_occlusion_attribution(
    model: Any,
    samples: np.ndarray,
    feature_names: list[str],
    baseline_window: np.ndarray,
) -> np.ndarray:
    base_pred = _predict_window_model(model, samples)
    attributions = np.zeros((len(samples), len(feature_names)), dtype=float)
    for feature_idx in range(len(feature_names)):
        occluded = samples.copy()
        occluded[:, :, feature_idx] = baseline_window[:, feature_idx]
        attributions[:, feature_idx] = base_pred - _predict_window_model(model, occluded)
    return attributions


def temporal_occlusion_attribution(
    model: Any,
    samples: np.ndarray,
    baseline_window: np.ndarray,
) -> np.ndarray:
    base_pred = _predict_window_model(model, samples)
    attributions = np.zeros((len(samples), samples.shape[1]), dtype=float)
    for time_idx in range(samples.shape[1]):
        occluded = samples.copy()
        occluded[:, time_idx, :] = baseline_window[time_idx, :]
        attributions[:, time_idx] = base_pred - _predict_window_model(model, occluded)
    return attributions


def lime_style_attribution(
    model: Any,
    samples: np.ndarray,
    feature_names: list[str],
    baseline_window: np.ndarray,
    random_state: int,
    perturbations: int = 96,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    attributions = np.zeros((len(samples), len(feature_names)), dtype=float)
    for sample_idx, sample in enumerate(samples):
        masks = rng.integers(0, 2, size=(perturbations, len(feature_names)))
        masks[0, :] = 1
        perturbed = np.repeat(sample[None, :, :], perturbations, axis=0)
        for row_idx, mask in enumerate(masks):
            off = np.where(mask == 0)[0]
            if off.size:
                perturbed[row_idx][:, off] = baseline_window[:, off]
        predictions = _predict_window_model(model, perturbed)
        distances = np.mean(1 - masks, axis=1)
        weights = np.exp(-(distances**2) / 0.25)
        surrogate = Ridge(alpha=0.01)
        surrogate.fit(masks, predictions, sample_weight=weights)
        attributions[sample_idx] = surrogate.coef_
    return attributions


def evaluate_explanation_quality(
    model: Any,
    bundle: WindowBundle,
    explanations: dict[str, Any],
    output_dir: Path,
) -> dict[str, float]:
    samples = explanations["samples"]
    sensor_attr = explanations["sensor_attr"]
    time_attr = explanations["time_attr"]
    lime_attr = explanations["lime_attr"]
    quality = {
        "faithfulness_deletion_delta": _faithfulness_deletion(
            model, samples, sensor_attr, bundle.baseline_window
        ),
        "stability_neighbor_cosine": _stability_neighbor(samples, sensor_attr),
        "consistency_occlusion_lime_spearman": _safe_spearman(
            np.mean(np.abs(sensor_attr), axis=0),
            np.mean(np.abs(lime_attr), axis=0),
        ),
        "robustness_noise_cosine": _robustness_noise(model, samples, sensor_attr, bundle),
        "sparsity_top3_share": _topk_share(sensor_attr, top_k=3),
        "localization_last20pct_share": _last_time_share(time_attr, share=0.2),
    }
    (output_dir / "xai_quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return quality


def evaluate_trustworthiness(
    models: dict[str, Any],
    best_name: str,
    bundle: WindowBundle,
    explanations: dict[str, Any],
    output_dir: Path,
) -> dict[str, float]:
    model_predictions = np.column_stack(
        [_predict_window_model(model, bundle.x_test) for model in models.values()]
    )
    best_pred = _predict_window_model(models[best_name], bundle.x_test)
    abs_error = np.abs(best_pred - bundle.y_test)
    uncertainty = np.std(model_predictions, axis=1)
    subgroup = _subgroup_fairness(bundle.meta_test, abs_error)
    top_attr_share = _low_variance_attribution_share(
        explanations["sensor_attr"],
        bundle.feature_names,
        bundle.low_variance_sensors,
    )
    trust = {
        "subgroup_fairness_mae_gap": subgroup["mae_gap"],
        "subgroup_fairness_max_group_mae": subgroup["max_group_mae"],
        "confidence_uncertainty_error_spearman": _safe_spearman(uncertainty, abs_error),
        "model_drift_wasserstein_mean": _mean_wasserstein(bundle.x_train, bundle.x_test),
        "adversarial_top_sensor_prediction_shift": _adversarial_shift(
            models[best_name],
            explanations["samples"],
            explanations["sensor_attr"],
            bundle.baseline_window,
        ),
        "spurious_low_variance_attribution_share": top_attr_share,
        "explanation_disagreement_one_minus_top5_jaccard": _explanation_disagreement(
            explanations["sensor_attr"],
            explanations["lime_attr"],
            bundle.feature_names,
        ),
    }
    (output_dir / "trustworthiness.json").write_text(
        json.dumps(trust, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return trust


def generate_decision_support(
    model: Any,
    bundle: WindowBundle,
    explanations: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    samples = explanations["samples"]
    predictions = _predict_window_model(model, samples)
    meta = explanations["meta"].copy()
    rows: list[dict[str, Any]] = []
    for idx, prediction in enumerate(predictions):
        top_sensors = _top_names(explanations["sensor_attr"][idx], bundle.feature_names, top_k=5)
        risk = "high" if prediction <= 20 else "medium" if prediction <= 50 else "low"
        window = (
            "next 15 to 20 cycles"
            if risk == "high"
            else "next 30 to 50 cycles"
            if risk == "medium"
            else "routine monitoring"
        )
        rows.append(
            {
                "unit_id": int(meta.loc[idx, "unit_id"]),
                "end_cycle": int(meta.loc[idx, "end_cycle"]),
                "actual_rul": float(meta.loc[idx, "target_rul"]),
                "predicted_rul": float(prediction),
                "risk_level": risk,
                "top_sensors": ", ".join(top_sensors),
                "maintenance_window": window,
                "decision_support_value": (
                    "prioritize sensor/component inspection using explanation evidence"
                ),
            }
        )
    advisor = pd.DataFrame(rows).sort_values(["risk_level", "predicted_rul"])
    advisor.to_csv(output_dir / "maintenance_advisor.csv", index=False)
    return advisor


def generate_prediction_intervals(
    models: dict[str, Any],
    bundle: WindowBundle,
    output_dir: Path,
) -> Path:
    predictions = {
        name: _predict_window_model(model, bundle.x_test) for name, model in models.items()
    }
    matrix = np.column_stack(list(predictions.values()))
    mean_pred = np.mean(matrix, axis=1)
    std_pred = np.std(matrix, axis=1)
    lower = mean_pred - 1.96 * std_pred
    upper = mean_pred + 1.96 * std_pred
    frame = bundle.meta_test[["unit_id", "start_cycle", "end_cycle", "target_rul"]].copy()
    for name, values in predictions.items():
        frame[f"pred_{name}"] = values
    frame["prediction_mean"] = mean_pred
    frame["prediction_std"] = std_pred
    frame["prediction_lower"] = lower
    frame["prediction_upper"] = upper
    frame["interval_covered"] = (bundle.y_test >= lower) & (bundle.y_test <= upper)
    path = output_dir / "prediction_intervals.csv"
    frame.to_csv(path, index=False)
    return path


def write_subgroup_metrics(model: Any, bundle: WindowBundle, output_dir: Path) -> Path:
    predictions = _predict_window_model(model, bundle.x_test)
    frame = bundle.meta_test[["unit_id", "start_cycle", "end_cycle", "target_rul"]].copy()
    frame["prediction"] = predictions
    frame["abs_error"] = np.abs(predictions - bundle.y_test)
    frame["squared_error"] = (predictions - bundle.y_test) ** 2
    frame["risk_group"] = _risk_groups(frame["target_rul"])
    rows = []
    for group, group_frame in frame.groupby("risk_group", observed=True):
        rows.append(
            {
                "risk_group": str(group),
                "count": int(len(group_frame)),
                "rmse": float(np.sqrt(np.mean(group_frame["squared_error"]))),
                "mae": float(np.mean(group_frame["abs_error"])),
                "mean_target_rul": float(np.mean(group_frame["target_rul"])),
                "mean_prediction": float(np.mean(group_frame["prediction"])),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("risk_group")
    path = output_dir / "subgroup_metrics.csv"
    metrics.to_csv(path, index=False)
    return path


def write_balanced_sampling_comparison(
    bundle: WindowBundle,
    output_dir: Path,
    run_config: XaiRunConfig,
) -> Path:
    rng = np.random.default_rng(run_config.random_state)
    original_indices = _sample_indices(len(bundle.y_train), run_config.max_train_windows, rng)
    balanced_indices = _balanced_sample_indices(bundle.y_train, run_config.max_train_windows, rng)
    variants = {
        "ridge_original_sampling": original_indices,
        "ridge_balanced_sampling": balanced_indices,
    }
    rows: list[dict[str, Any]] = []
    for variant, indices in variants.items():
        model = Ridge(alpha=1.0)
        model.fit(_flatten(bundle.x_train[indices]), bundle.y_train[indices])
        predictions = model.predict(_flatten(bundle.x_test))
        frame = bundle.meta_test[["target_rul"]].copy()
        frame["prediction"] = predictions
        frame["abs_error"] = np.abs(predictions - bundle.y_test)
        frame["squared_error"] = (predictions - bundle.y_test) ** 2
        frame["risk_group"] = _risk_groups(frame["target_rul"])
        rows.append(
            {
                "variant": variant,
                "risk_group": "overall",
                "train_windows": int(len(indices)),
                "rmse": float(np.sqrt(np.mean(frame["squared_error"]))),
                "mae": float(np.mean(frame["abs_error"])),
            }
        )
        for group, group_frame in frame.groupby("risk_group", observed=True):
            rows.append(
                {
                    "variant": variant,
                    "risk_group": str(group),
                    "train_windows": int(len(indices)),
                    "rmse": float(np.sqrt(np.mean(group_frame["squared_error"]))),
                    "mae": float(np.mean(group_frame["abs_error"])),
                }
            )
    metrics = pd.DataFrame(rows).sort_values(["variant", "risk_group"])
    path = output_dir / "fairness_mitigation.csv"
    metrics.to_csv(path, index=False)
    return path


def run_tuning_grid(bundle: WindowBundle, output_dir: Path, run_config: XaiRunConfig) -> Path:
    configs = [
        ("lstm_sequence", "lstm", 32, 1e-3, 0.1),
        ("gru_sequence", "gru", 32, 1e-3, 0.1),
        ("transformer_sequence", "transformer", 48, 1e-3, 0.1),
    ]
    rows = []
    rng = np.random.default_rng(run_config.random_state)
    size = min(len(bundle.x_train), run_config.max_train_windows)
    indices = rng.choice(len(bundle.x_train), size=size, replace=False)
    for name, model_type, hidden_size, learning_rate, dropout in configs:
        config = TorchTrainingConfig(
            epochs=max(2, min(run_config.deep_epochs, 6)),
            batch_size=run_config.deep_batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            dropout=dropout,
            random_state=run_config.random_state,
        )
        model = TorchSequenceRegressor(model_type, bundle.x_train.shape[1:], config)
        model.fit(bundle.x_train[indices], bundle.y_train[indices])
        validation_pred = _predict_window_model(model, bundle.x_validation)
        test_pred = _predict_window_model(model, bundle.x_test)
        rows.append(
            {
                "model": name,
                "epochs": config.epochs,
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "dropout": dropout,
                "validation_rmse": rmse(bundle.y_validation, validation_pred),
                "validation_mae": mae(bundle.y_validation, validation_pred),
                "test_rmse": rmse(bundle.y_test, test_pred),
                "test_mae": mae(bundle.y_test, test_pred),
            }
        )
    frame = pd.DataFrame(rows).sort_values("validation_rmse")
    output_path = output_dir / "tuning_grid.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    frame.to_csv(experiments_dir / f"tuning_{output_dir.name}.csv", index=False)
    return output_path


def build_dimension_framework(
    model_metrics: pd.DataFrame,
    quality: dict[str, float],
    trust: dict[str, float],
    advisor: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    best = model_metrics.iloc[0]
    rows = [
        (
            "Predictive performance",
            "Is the AI accurate?",
            f"Best model {best['model']} test RMSE {best['test_rmse']:.3f}.",
        ),
        (
            "Interpretability",
            "Can we understand why it predicted?",
            "Global sensor, local sensor, temporal, and counterfactual explanations generated.",
        ),
        (
            "Faithfulness",
            "Does explanation represent model behavior?",
            f"Deletion delta {quality['faithfulness_deletion_delta']:.3f}.",
        ),
        (
            "Stability",
            "Do similar inputs produce similar explanations?",
            f"Neighbor cosine {quality['stability_neighbor_cosine']:.3f}.",
        ),
        (
            "Robustness",
            "Does explanation survive perturbations?",
            f"Noise cosine {quality['robustness_noise_cosine']:.3f}.",
        ),
        (
            "Fairness",
            "Are decisions consistent across groups?",
            f"RUL subgroup MAE gap {trust['subgroup_fairness_mae_gap']:.3f}.",
        ),
        (
            "Confidence",
            "Does reliability match confidence?",
            f"Uncertainty-error Spearman {trust['confidence_uncertainty_error_spearman']:.3f}.",
        ),
        (
            "Actionability",
            "Can explanation suggest what to do?",
            f"{len(advisor)} maintenance-advisor rows generated.",
        ),
        (
            "Human usefulness",
            "Does explanation improve decisions?",
            "Engineer can inspect top sensors and high-risk engines first.",
        ),
        (
            "Domain impact",
            "Does XAI produce practical benefit?",
            (
                "Maintenance prioritization combines RUL, risk, attribution, "
                "and counterfactual evidence."
            ),
        ),
    ]
    frame = pd.DataFrame(rows, columns=["dimension", "question", "project_impact"])
    frame.to_csv(output_dir / "xai_evaluation_framework.csv", index=False)
    return frame


def nearest_counterfactuals(
    model: Any,
    samples: np.ndarray,
    bundle: WindowBundle,
    sample_indices: np.ndarray,
    min_improvement: float = 10.0,
) -> pd.DataFrame:
    train_flat = _flatten(bundle.x_train)
    train_pred = _predict_window_model(model, bundle.x_train)
    sample_pred = _predict_window_model(model, samples)
    rows: list[dict[str, Any]] = []
    for local_idx, prediction in enumerate(sample_pred):
        candidates = np.where(train_pred >= prediction + min_improvement)[0]
        if len(candidates) == 0:
            candidates = np.argsort(train_pred)[-25:]
        sample_flat = samples[local_idx].reshape(1, -1)
        distances = np.linalg.norm(train_flat[candidates] - sample_flat, axis=1)
        nearest_idx = int(candidates[int(np.argmin(distances))])
        delta = bundle.x_train[nearest_idx] - samples[local_idx]
        sensor_delta = np.mean(delta, axis=0)
        top_change = _top_names(sensor_delta, bundle.feature_names, top_k=5)
        rows.append(
            {
                "sample_index": int(sample_indices[local_idx]),
                "current_prediction": float(prediction),
                "counterfactual_prediction": float(train_pred[nearest_idx]),
                "expected_rul_gain": float(train_pred[nearest_idx] - prediction),
                "nearest_train_unit": int(bundle.meta_train.loc[nearest_idx, "unit_id"]),
                "changed_sensors": ", ".join(top_change),
                "interpretation": (
                    "nearest healthier historical window suggests sensor pattern to inspect"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_xai_report(
    report_dir: Path,
    dataset_config: DatasetConfig,
    run_config: XaiRunConfig,
    model_metrics: pd.DataFrame,
    best_name: str,
    quality: dict[str, float],
    trust: dict[str, float],
    framework: pd.DataFrame,
) -> Path:
    report_path = report_dir / "xai_evaluation_report.md"
    lines = [
        "# XAI evaluation report",
        "",
        f"Subset: {dataset_config.subset}",
        f"Window length: {run_config.window_length}",
        f"Best model: {best_name}",
        "",
        "## Predictive benchmark",
        "",
        _frame_table(model_metrics),
        "",
        "## Explanation quality",
        "",
        _dict_table(quality),
        "",
        "## Trustworthy behavior checks",
        "",
        _dict_table(trust),
        "",
        "## XAI evaluation framework",
        "",
        _frame_table(framework),
        "",
        "## Run scope",
        "",
        (
            f"Deep sequence models enabled: {run_config.include_deep_models}. "
            f"Package-backed SHAP/LIME/DiCE enabled: {run_config.exact_xai}."
        ),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _load_or_prepare(
    dataset_config: DatasetConfig, preprocessing_config: PreprocessingConfig
) -> Any:
    output_dir = dataset_config.processed_dir / dataset_config.subset
    required = [
        output_dir / "train.parquet",
        output_dir / "test.parquet",
        output_dir / "windows.npz",
        output_dir / "window_metadata.parquet",
    ]
    if all(path.exists() for path in required):
        return load_prepared_subset(dataset_config)
    return prepare_subset(dataset_config, preprocessing_config)


def _flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def _expects_sequence_input(model: Any) -> bool:
    return bool(getattr(model, "expects_sequence_input", False))


def _predict_window_model(model: Any, x: np.ndarray) -> np.ndarray:
    if _expects_sequence_input(model):
        return np.asarray(model.predict(x), dtype=float)
    return np.asarray(model.predict(_flatten(x)), dtype=float)


def _timed(
    name: str,
    call: Any,
    rows: list[dict[str, Any]],
    enabled: bool,
    sample_count: int,
) -> Any:
    if not enabled:
        return call()
    tracemalloc.start()
    start = time.perf_counter()
    status = "success"
    result: Any = None
    try:
        result = call()
        return result
    except Exception as exc:
        status = f"failed: {exc.__class__.__name__}"
        raise
    finally:
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(
            {
                "explainer": name,
                "seconds": elapsed,
                "seconds_per_sample": elapsed / sample_count if sample_count else np.nan,
                "peak_memory_mb": peak / (1024 * 1024),
                "sample_count": sample_count,
                "output_shape": _shape_text(result),
                "status": status,
            }
        )


def _shape_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return "x".join(str(part) for part in value.shape)
    if isinstance(value, pd.DataFrame):
        return f"{len(value)}x{len(value.columns)}"
    if isinstance(value, dict):
        return ",".join(value.keys())
    return type(value).__name__


def _select_xai_samples(y: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(y)
    if len(order) <= count:
        return order
    positions = np.linspace(0, len(order) - 1, count).astype(int)
    return order[positions]


def _sample_indices(length: int, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return rng.choice(length, size=maximum, replace=False)


def _balanced_sample_indices(
    y: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = _risk_groups(pd.Series(y))
    group_indices = [np.flatnonzero(groups.to_numpy() == group) for group in groups.cat.categories]
    group_indices = [indices for indices in group_indices if len(indices) > 0]
    if not group_indices:
        return _sample_indices(len(y), maximum, rng)
    per_group = max(1, maximum // len(group_indices))
    sampled = [
        rng.choice(indices, size=per_group, replace=len(indices) < per_group)
        for indices in group_indices
    ]
    result = np.concatenate(sampled)
    rng.shuffle(result)
    return result[:maximum]


def _global_importance_frame(
    attr: np.ndarray, feature_names: list[str], method: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_attribution": np.mean(attr, axis=0),
            "mean_abs_attribution": np.mean(np.abs(attr), axis=0),
            "method": method,
        }
    ).sort_values("mean_abs_attribution", ascending=False)


def _top_names(values: np.ndarray, names: list[str], top_k: int) -> list[str]:
    indices = np.argsort(np.abs(values))[-top_k:][::-1]
    return [names[int(idx)] for idx in indices]


def _faithfulness_deletion(
    model: Any,
    samples: np.ndarray,
    sensor_attr: np.ndarray,
    baseline_window: np.ndarray,
    top_k: int = 3,
) -> float:
    original = _predict_window_model(model, samples)
    occluded = samples.copy()
    for row_idx in range(len(samples)):
        top = np.argsort(np.abs(sensor_attr[row_idx]))[-top_k:]
        occluded[row_idx][:, top] = baseline_window[:, top]
    changed = _predict_window_model(model, occluded)
    return float(np.mean(np.abs(original - changed)))


def _stability_neighbor(samples: np.ndarray, sensor_attr: np.ndarray) -> float:
    flat = _flatten(samples)
    similarities = cosine_similarity(flat)
    np.fill_diagonal(similarities, -np.inf)
    nearest = np.argmax(similarities, axis=1)
    return float(
        np.mean([_cosine(sensor_attr[i], sensor_attr[nearest[i]]) for i in range(len(samples))])
    )


def _robustness_noise(
    model: Any,
    samples: np.ndarray,
    sensor_attr: np.ndarray,
    bundle: WindowBundle,
    sigma: float = 0.05,
) -> float:
    rng = np.random.default_rng(123)
    noisy = samples + rng.normal(0.0, sigma, size=samples.shape)
    noisy_attr = sensor_occlusion_attribution(
        model, noisy, bundle.feature_names, bundle.baseline_window
    )
    return float(np.mean([_cosine(sensor_attr[i], noisy_attr[i]) for i in range(len(samples))]))


def _topk_share(attr: np.ndarray, top_k: int) -> float:
    shares = []
    for row in np.abs(attr):
        total = float(np.sum(row))
        shares.append(float(np.sum(np.sort(row)[-top_k:]) / total) if total else 0.0)
    return float(np.mean(shares))


def _last_time_share(time_attr: np.ndarray, share: float) -> float:
    start = int(time_attr.shape[1] * (1 - share))
    abs_attr = np.abs(time_attr)
    totals = np.sum(abs_attr, axis=1)
    localized = np.sum(abs_attr[:, start:], axis=1)
    ratios = np.divide(localized, totals, out=np.zeros_like(localized), where=totals > 0)
    return float(np.mean(ratios))


def _subgroup_fairness(meta: pd.DataFrame, abs_error: np.ndarray) -> dict[str, float]:
    frame = meta.copy()
    frame["abs_error"] = abs_error
    frame["risk_group"] = _risk_groups(frame["target_rul"])
    group_mae = frame.groupby("risk_group", observed=True)["abs_error"].mean()
    return {
        "mae_gap": float(group_mae.max() - group_mae.min()),
        "max_group_mae": float(group_mae.max()),
    }


def _risk_groups(target_rul: pd.Series) -> pd.Series:
    return pd.cut(
        target_rul,
        bins=[-np.inf, 30, 80, np.inf],
        labels=["high_risk", "medium_risk", "low_risk"],
    )


def _mean_wasserstein(x_train: np.ndarray, x_test: np.ndarray) -> float:
    train_mean = np.mean(x_train, axis=1)
    test_mean = np.mean(x_test, axis=1)
    distances = [
        wasserstein_distance(train_mean[:, feature_idx], test_mean[:, feature_idx])
        for feature_idx in range(train_mean.shape[1])
    ]
    return float(np.mean(distances))


def _adversarial_shift(
    model: Any,
    samples: np.ndarray,
    sensor_attr: np.ndarray,
    baseline_window: np.ndarray,
    epsilon: float = 0.2,
) -> float:
    original = _predict_window_model(model, samples)
    perturbed = samples.copy()
    for row_idx in range(len(samples)):
        top = np.argsort(np.abs(sensor_attr[row_idx]))[-3:]
        direction = np.sign(samples[row_idx][:, top] - baseline_window[:, top])
        perturbed[row_idx][:, top] += epsilon * np.where(direction == 0, 1, direction)
    changed = _predict_window_model(model, perturbed)
    return float(np.mean(np.abs(original - changed)))


def _low_variance_attribution_share(
    sensor_attr: np.ndarray,
    feature_names: list[str],
    low_variance_sensors: list[str],
) -> float:
    if not low_variance_sensors:
        return 0.0
    indices = [
        feature_names.index(sensor) for sensor in low_variance_sensors if sensor in feature_names
    ]
    abs_attr = np.abs(sensor_attr)
    total = float(np.sum(abs_attr))
    return float(np.sum(abs_attr[:, indices]) / total) if total else 0.0


def _explanation_disagreement(
    sensor_attr: np.ndarray,
    lime_attr: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
) -> float:
    del feature_names
    scores = []
    for row_a, row_b in zip(sensor_attr, lime_attr, strict=True):
        top_a = set(np.argsort(np.abs(row_a))[-top_k:].tolist())
        top_b = set(np.argsort(np.abs(row_b))[-top_k:].tolist())
        scores.append(1.0 - len(top_a & top_b) / len(top_a | top_b))
    return float(np.mean(scores))


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    correlation = spearmanr(a, b).correlation
    return float(correlation) if correlation is not None and np.isfinite(correlation) else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _dict_table(values: dict[str, float]) -> str:
    rows = ["| Metric | Value |", "| --- | ---: |"]
    for key, value in values.items():
        rows.append(f"| {key} | {value:.4f} |")
    return "\n".join(rows)


def _frame_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        rendered = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join(rows)
