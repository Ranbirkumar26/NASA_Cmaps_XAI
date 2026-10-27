from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cmapss_xai.models.sequence import TorchSequenceRegressor, TorchTrainingConfig
from cmapss_xai.xai.integrated_gradients import integrated_gradients_attribution
from cmapss_xai.xai.package_backed import (
    PackageBackedXaiConfig,
    flat_feature_names,
    lime_sensor_attribution,
)
from cmapss_xai.xai.project import (
    WindowBundle,
    XaiRunConfig,
    _timed,
    generate_prediction_intervals,
    write_balanced_sampling_comparison,
    write_subgroup_metrics,
)


class MeanModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float).mean(axis=1)


def test_flat_feature_names_preserve_time_then_feature_order() -> None:
    assert flat_feature_names(2, ["sensor_1", "sensor_2"]) == [
        "t00_sensor_1",
        "t00_sensor_2",
        "t01_sensor_1",
        "t01_sensor_2",
    ]


def test_torch_sequence_regressor_predicts_tiny_lstm() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 3, 2)).astype(np.float32)
    y = x[:, -1, :].sum(axis=1)
    model = TorchSequenceRegressor(
        model_type="lstm",
        window_shape=(3, 2),
        config=TorchTrainingConfig(epochs=1, batch_size=8, hidden_size=16),
    )
    model.fit(x, y)
    predictions = model.predict(x[:4])
    assert predictions.shape == (4,)
    assert np.all(np.isfinite(predictions))


def test_torch_sequence_regressor_predicts_tiny_gru() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    x = rng.normal(size=(16, 3, 2)).astype(np.float32)
    y = x[:, -1, :].sum(axis=1)
    model = TorchSequenceRegressor(
        model_type="gru",
        window_shape=(3, 2),
        config=TorchTrainingConfig(epochs=1, batch_size=8, hidden_size=16),
    )
    model.fit(x, y)
    predictions = model.predict(x[:4])
    assert predictions.shape == (4,)
    assert np.all(np.isfinite(predictions))


def test_integrated_gradients_returns_sensor_and_temporal_shapes() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("captum")
    rng = np.random.default_rng(13)
    x = rng.normal(size=(10, 3, 2)).astype(np.float32)
    y = x[:, -1, :].sum(axis=1)
    model = TorchSequenceRegressor(
        model_type="gru",
        window_shape=(3, 2),
        config=TorchTrainingConfig(epochs=1, batch_size=5, hidden_size=16),
    )
    model.fit(x, y)
    meta = _metadata(len(x[:2]))
    result = integrated_gradients_attribution(
        model,
        x[:2],
        np.zeros((3, 2), dtype=np.float32),
        ["a", "b"],
        meta,
        n_steps=4,
    )
    assert result.sensor_attribution.shape == (2, 2)
    assert result.temporal_attribution.shape == (2, 3)
    assert np.all(np.isfinite(result.sensor_attribution))


def test_lime_package_backend_returns_sensor_attribution_shape() -> None:
    pytest.importorskip("lime")
    rng = np.random.default_rng(9)
    background = rng.normal(size=(12, 2, 3))
    samples = rng.normal(size=(2, 2, 3))
    attr = lime_sensor_attribution(
        MeanModel(),
        background,
        samples,
        ["a", "b", "c"],
        PackageBackedXaiConfig(background_size=8, lime_samples=2),
    )
    assert attr.shape == (2, 3)
    assert np.all(np.isfinite(attr))


def test_prediction_intervals_and_subgroup_metrics_have_expected_schema(tmp_path) -> None:
    bundle = WindowBundle(
        x_train=np.zeros((2, 3, 2)),
        y_train=np.array([10.0, 20.0]),
        meta_train=_metadata(2),
        x_validation=np.zeros((2, 3, 2)),
        y_validation=np.array([10.0, 20.0]),
        meta_validation=_metadata(2),
        x_test=np.zeros((3, 3, 2)),
        y_test=np.array([10.0, 50.0, 100.0]),
        meta_test=_metadata(3, targets=[10.0, 50.0, 100.0]),
        feature_names=["a", "b"],
        baseline_window=np.zeros((3, 2)),
        low_variance_sensors=[],
    )
    interval_path = generate_prediction_intervals(
        {"low": ConstantModel(20.0), "high": ConstantModel(80.0)},
        bundle,
        tmp_path,
    )
    subgroup_path = write_subgroup_metrics(ConstantModel(40.0), bundle, tmp_path)
    mitigation_path = write_balanced_sampling_comparison(
        bundle,
        tmp_path,
        XaiRunConfig(max_train_windows=6),
    )
    interval_columns = set(np.genfromtxt(interval_path, delimiter=",", dtype=str, max_rows=1))
    subgroup_columns = set(np.genfromtxt(subgroup_path, delimiter=",", dtype=str, max_rows=1))
    mitigation_columns = set(
        np.genfromtxt(mitigation_path, delimiter=",", dtype=str, max_rows=1)
    )
    assert {"prediction_lower", "prediction_upper", "interval_covered"} <= interval_columns
    assert {"risk_group", "count", "rmse", "mae"} <= subgroup_columns
    assert {"variant", "risk_group", "train_windows", "rmse", "mae"} <= mitigation_columns


def test_runtime_benchmark_row_records_success_shape_and_memory() -> None:
    rows = []
    result = _timed("toy_explainer", lambda: np.zeros((2, 3)), rows, True, 2)
    assert result.shape == (2, 3)
    assert rows[0]["explainer"] == "toy_explainer"
    assert rows[0]["status"] == "success"
    assert rows[0]["output_shape"] == "2x3"
    assert rows[0]["seconds_per_sample"] >= 0.0
    assert rows[0]["peak_memory_mb"] >= 0.0


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


def _metadata(count: int, targets: list[float] | None = None) -> pd.DataFrame:
    values = targets or [float(index + 1) for index in range(count)]
    return pd.DataFrame(
        {
            "sample_index": list(range(count)),
            "unit_id": [1] * count,
            "start_cycle": list(range(1, count + 1)),
            "end_cycle": list(range(30, 30 + count)),
            "target_rul": values,
        }
    )
