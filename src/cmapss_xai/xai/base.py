from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Explanation:
    method: str
    values: np.ndarray | pd.DataFrame
    metadata: dict[str, Any]


class BaseExplainer(ABC):
    name: str

    @abstractmethod
    def explain(self, inputs: np.ndarray, **kwargs: Any) -> Explanation:
        raise NotImplementedError

    @abstractmethod
    def explain_global(self, inputs: np.ndarray, **kwargs: Any) -> Explanation:
        raise NotImplementedError

    @abstractmethod
    def explain_local(self, inputs: np.ndarray, index: int, **kwargs: Any) -> Explanation:
        raise NotImplementedError
