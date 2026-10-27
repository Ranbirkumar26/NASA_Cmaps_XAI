from __future__ import annotations

import pandas as pd


def compute_train_rul(df: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    frame = df.copy()
    max_cycle = frame.groupby("unit_id")["cycle"].transform("max")
    frame["rul"] = max_cycle - frame["cycle"]
    frame["rul_capped"] = frame["rul"].clip(upper=cap) if cap is not None else frame["rul"]
    return frame


def attach_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    frame = test_df.copy()
    max_cycle = frame.groupby("unit_id")["cycle"].transform("max")
    final_rul = rul_df.set_index("unit_id")["final_rul"]
    frame["final_rul"] = frame["unit_id"].map(final_rul)
    if frame["final_rul"].isna().any():
        missing = sorted(frame.loc[frame["final_rul"].isna(), "unit_id"].unique().tolist())
        raise ValueError(f"Missing final RUL labels for test units: {missing}")
    frame["rul"] = (max_cycle - frame["cycle"]) + frame["final_rul"]
    frame["rul_capped"] = frame["rul"].clip(upper=cap) if cap is not None else frame["rul"]
    return frame
