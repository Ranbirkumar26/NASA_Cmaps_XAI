from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntegratedGradientsResult:
    sensor_attribution: np.ndarray
    temporal_attribution: np.ndarray
    local_examples: pd.DataFrame


def integrated_gradients_attribution(
    model: Any,
    samples: np.ndarray,
    baseline_window: np.ndarray,
    feature_names: list[str],
    metadata: pd.DataFrame,
    n_steps: int = 24,
) -> IntegratedGradientsResult:
    if not getattr(model, "expects_sequence_input", False) or getattr(model, "model", None) is None:
        raise ValueError("Integrated Gradients requires a fitted torch sequence model")
    captum_attr = _require_captum()
    torch = _require_torch()

    network = model.model
    network.eval()

    def forward(inputs: Any) -> Any:
        outputs = network(inputs)
        return outputs * model.target_scale_ + model.target_mean_

    explainer = captum_attr.IntegratedGradients(forward)
    x = torch.tensor(samples.astype(np.float32), requires_grad=True)
    baseline = np.repeat(baseline_window[None, :, :], len(samples), axis=0).astype(np.float32)
    baseline_tensor = torch.tensor(baseline)
    attributions = explainer.attribute(x, baselines=baseline_tensor, n_steps=n_steps)
    values = attributions.detach().cpu().numpy()
    sensor_attr = np.sum(values, axis=1)
    temporal_attr = np.sum(np.abs(values), axis=2)
    rows = []
    for idx, row in metadata.reset_index(drop=True).iterrows():
        rows.append(
            {
                "sample_index": int(row.get("sample_index", idx)),
                "unit_id": int(row["unit_id"]),
                "start_cycle": int(row["start_cycle"]),
                "end_cycle": int(row["end_cycle"]),
                "target_rul": float(row["target_rul"]),
                "top_sensors": ", ".join(_top_names(sensor_attr[idx], feature_names, 5)),
                "top_time_index": int(np.argmax(np.abs(temporal_attr[idx]))),
            }
        )
    return IntegratedGradientsResult(
        sensor_attribution=sensor_attr,
        temporal_attribution=temporal_attr,
        local_examples=pd.DataFrame(rows),
    )


def _top_names(values: np.ndarray, names: list[str], top_k: int) -> list[str]:
    indices = np.argsort(np.abs(values))[-top_k:][::-1]
    return [names[int(idx)] for idx in indices]


def _require_captum() -> Any:
    try:
        from captum import attr
    except ImportError as exc:
        raise RuntimeError(
            "Integrated Gradients requires installing the deep-xai extra: "
            'python3.11 -m pip install -e ".[deep-xai]"'
        ) from exc
    return attr


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Integrated Gradients requires installing the deep-xai extra: "
            'python3.11 -m pip install -e ".[deep-xai]"'
        ) from exc
    return torch
