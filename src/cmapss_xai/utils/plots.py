from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def engine_sensor_trends(frame: pd.DataFrame, sensors: list[str]) -> Figure:
    long = frame.melt(
        id_vars=["cycle"],
        value_vars=sensors,
        var_name="sensor",
        value_name="value",
    )
    return px.line(long, x="cycle", y="value", color="sensor")


def rul_curve(frame: pd.DataFrame) -> Figure:
    return px.line(frame, x="cycle", y="rul_capped", title="RUL curve")


def sensor_distribution(frame: pd.DataFrame, sensor: str) -> Figure:
    return px.histogram(frame, x=sensor, nbins=50, title=f"{sensor} distribution")


def sensor_unit_trends(frame: pd.DataFrame, sensor: str) -> Figure:
    return px.line(frame, x="cycle", y=sensor, color="unit_id", title=f"{sensor} trends")


def operating_condition_scatter(frame: pd.DataFrame, settings: list[str]) -> Figure:
    return px.scatter(frame, x=settings[0], y=settings[1], color="split")


def model_rmse_bar(metrics: pd.DataFrame) -> Figure:
    return px.bar(metrics, x="model", y="test_rmse", title="Test RMSE")


def subgroup_mae_bar(subgroups: pd.DataFrame) -> Figure:
    return px.bar(subgroups, x="risk_group", y="mae", title="Subgroup MAE")


def sensor_importance_bar(frame: pd.DataFrame, limit: int = 12) -> Figure:
    return px.bar(frame.head(limit), x="feature", y="mean_abs_attribution")


def temporal_importance_bar(frame: pd.DataFrame) -> Figure:
    return px.bar(frame, x="time_index", y="mean_abs_attribution")


def explainer_runtime_bar(runtime: pd.DataFrame) -> Figure:
    return px.bar(runtime, x="explainer", y="seconds_per_sample")


def risk_distribution(advisor: pd.DataFrame) -> Figure:
    return px.histogram(advisor, x="risk_level", title="Risk distribution")
