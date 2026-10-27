from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _as_arrays(y_true, y_pred)
    residual = np.sum((true - pred) ** 2)
    total = np.sum((true - np.mean(true)) ** 2)
    if total == 0:
        return 0.0
    return float(1 - residual / total)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true, pred = _as_arrays(y_true, y_pred)
    diff = pred - true
    penalties = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(penalties))


def _as_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    if true.size == 0:
        raise ValueError("Metric inputs must not be empty")
    return true, pred
