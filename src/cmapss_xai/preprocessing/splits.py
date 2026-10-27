from __future__ import annotations

import numpy as np
import pandas as pd


def assign_unit_splits(
    df: pd.DataFrame,
    validation_unit_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not 0 <= validation_unit_fraction < 1:
        raise ValueError("validation_unit_fraction must be in [0, 1)")
    frame = df.copy()
    units = np.array(sorted(frame["unit_id"].unique().tolist()))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    validation_count = int(round(len(units) * validation_unit_fraction))
    validation_units = set(shuffled[:validation_count].tolist())
    frame["split"] = np.where(frame["unit_id"].isin(validation_units), "validation", "train")
    return frame
