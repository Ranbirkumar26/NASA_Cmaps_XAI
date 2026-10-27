from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CmapssDataset:
    subset: str
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame
    processed_train: pd.DataFrame
    processed_test: pd.DataFrame
    test_rul: pd.DataFrame
    metadata: dict[str, Any]
    scaler: StandardScaler
    feature_registry: dict[str, Any]
    window_arrays: dict[str, np.ndarray]
    window_metadata: pd.DataFrame
    feature_names: list[str]
    sensor_statistics: pd.DataFrame
