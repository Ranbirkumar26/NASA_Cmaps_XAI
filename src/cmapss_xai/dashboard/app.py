from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from cmapss_xai.utils import plots

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    st.set_page_config(page_title="C-MAPSS XAI Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
        }
        [data-testid="stMetric"] * {
            color: #0f172a !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
            line-height: 1.2 !important;
        }
        [data-testid="stMetricLabel"] {
            white-space: nowrap;
        }
        [data-testid="stMetricLabel"] * {
            color: #475569 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("C-MAPSS XAI Dashboard")
    subset = st.sidebar.selectbox("Subset", available_subsets(), index=0)
    page = st.sidebar.radio(
        "Page",
        [
            "Final review overview",
            "Engine explorer",
            "Sensor explorer",
            "Operating conditions",
            "Model benchmark",
            "XAI explanations",
            "Counterfactuals",
            "Maintenance advisor",
            "Report viewer",
        ],
    )
    paths = artifact_paths(subset)
    if page == "Final review overview":
        final_review_overview(paths, subset)
    elif page == "Engine explorer":
        engine_explorer(paths)
    elif page == "Sensor explorer":
        sensor_explorer(paths)
    elif page == "Operating conditions":
        operating_conditions(paths)
    elif page == "Model benchmark":
        model_benchmark(paths)
    elif page == "XAI explanations":
        xai_explanations(paths)
    elif page == "Counterfactuals":
        counterfactuals(paths)
    elif page == "Maintenance advisor":
        maintenance_advisor(paths)
    else:
        report_viewer(paths)


def available_subsets() -> list[str]:
    processed = ROOT / "data" / "processed"
    subsets = sorted(path.name for path in processed.glob("FD*") if path.is_dir())
    return subsets or ["FD001"]


def artifact_paths(subset: str) -> dict[str, Path]:
    return {
        "processed": ROOT / "data" / "processed" / subset,
        "results": ROOT / "results" / "xai" / subset,
        "cross_subset": ROOT / "results" / "xai" / "cross_subset_benchmark.csv",
        "report": ROOT / "reports" / "xai_evaluation" / subset / "xai_evaluation_report.md",
    }


def final_review_overview(paths: dict[str, Path], subset: str) -> None:
    metrics = read_csv(paths["results"] / "model_benchmark.csv")
    framework = read_csv(paths["results"] / "xai_evaluation_framework.csv")
    runtime = read_csv(paths["results"] / "explainer_runtime.csv")
    advisor = read_csv(paths["results"] / "maintenance_advisor.csv")
    cross_subset = read_csv(paths["cross_subset"])
    quality = read_json(paths["results"] / "xai_quality.json")
    trust = read_json(paths["results"] / "trustworthiness.json")

    st.subheader("Final review snapshot")
    st.caption(
        "Cached dashboard for presentation: model results, XAI evidence, trust checks, "
        "counterfactuals, and maintenance-advisor output. No model training runs inside Streamlit."
    )

    best = best_model_row(metrics)
    cols = st.columns(4)
    cols[0].metric("Subset", subset)
    cols[1].metric("Best model", display_model_name(best.get("model", "missing")))
    cols[2].metric("Test RMSE", format_number(best.get("test_rmse")))
    cols[3].metric("Test MAE", format_number(best.get("test_mae")))

    cols = st.columns(4)
    cols[0].metric("XAI methods checked", str(len(runtime)) if not runtime.empty else "0")
    cols[1].metric("Advisor cases", str(len(advisor)) if not advisor.empty else "0")
    cols[2].metric("Faithfulness", format_number(quality.get("faithfulness_deletion_delta")))
    cols[3].metric("Robustness", format_number(quality.get("robustness_noise_cosine")))

    left, right = st.columns([1.1, 0.9])
    with left:
        if not metrics.empty:
            st.plotly_chart(plots.model_rmse_bar(metrics), width="stretch")
    with right:
        if framework.empty:
            st.info("XAI evaluation framework not found.")
        else:
            st.dataframe(framework, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        if not cross_subset.empty:
            st.subheader("Cross-subset smoke benchmark")
            st.dataframe(cross_subset, width="stretch", hide_index=True)
    with right:
        st.subheader("Trust status")
        trust_rows = [
            (
                "Uncertainty-error Spearman",
                trust.get("confidence_uncertainty_error_spearman"),
            ),
            ("Subgroup MAE gap", trust.get("subgroup_fairness_mae_gap")),
            (
                "Explanation disagreement",
                trust.get("explanation_disagreement_one_minus_top5_jaccard"),
            ),
            (
                "Low-variance attribution share",
                trust.get("spurious_low_variance_attribution_share"),
            ),
        ]
        st.dataframe(
            pd.DataFrame(
                {
                    "check": [row[0] for row in trust_rows],
                    "value": [format_number(row[1]) for row in trust_rows],
                }
            ),
            width="stretch",
            hide_index=True,
        )


def engine_explorer(paths: dict[str, Path]) -> None:
    test = read_table(paths["processed"] / "test.parquet")
    if test.empty:
        st.warning("Processed test data not found.")
        return
    unit = st.selectbox("Engine", sorted(test["unit_id"].unique()))
    sensors = [column for column in test.columns if column.startswith("sensor_")]
    selected = st.multiselect("Sensors", sensors, default=sensors[:3])
    frame = test[test["unit_id"] == unit]
    if selected:
        st.plotly_chart(
            plots.engine_sensor_trends(frame, selected),
            width="stretch",
        )
    if "rul_capped" in frame:
        st.plotly_chart(
            plots.rul_curve(frame),
            width="stretch",
        )


def sensor_explorer(paths: dict[str, Path]) -> None:
    train = read_table(paths["processed"] / "train.parquet")
    report = read_csv(paths["processed"] / "sensor_report.csv")
    if train.empty:
        st.warning("Processed train data not found.")
        return
    sensors = [column for column in train.columns if column.startswith("sensor_")]
    sensor = st.selectbox("Sensor", sensors)
    st.plotly_chart(
        plots.sensor_distribution(train, sensor),
        width="stretch",
    )
    st.plotly_chart(
        plots.sensor_unit_trends(train, sensor),
        width="stretch",
    )
    if not report.empty:
        st.dataframe(report[report["sensor"] == sensor], width="stretch")


def operating_conditions(paths: dict[str, Path]) -> None:
    train = read_table(paths["processed"] / "train.parquet")
    if train.empty:
        st.warning("Processed train data not found.")
        return
    settings = [column for column in train.columns if column.startswith("setting_")]
    st.dataframe(train[settings].describe().T, width="stretch")
    if len(settings) >= 2:
        st.plotly_chart(
            plots.operating_condition_scatter(train, settings),
            width="stretch",
        )


def model_benchmark(paths: dict[str, Path]) -> None:
    metrics = read_csv(paths["results"] / "model_benchmark.csv")
    intervals = read_csv(paths["results"] / "prediction_intervals.csv")
    subgroups = read_csv(paths["results"] / "subgroup_metrics.csv")
    if metrics.empty:
        st.warning("Model benchmark not found.")
        return
    st.dataframe(metrics, width="stretch")
    st.plotly_chart(
        plots.model_rmse_bar(metrics),
        width="stretch",
    )
    if not intervals.empty:
        coverage = float(intervals["interval_covered"].mean())
        st.metric("Interval coverage", f"{coverage:.3f}")
    if not subgroups.empty:
        st.plotly_chart(
            plots.subgroup_mae_bar(subgroups),
            width="stretch",
        )


def xai_explanations(paths: dict[str, Path]) -> None:
    explanation_dir = paths["results"] / "explanations"
    sensor = read_csv(explanation_dir / "global_sensor_importance.csv")
    temporal = read_csv(explanation_dir / "global_temporal_importance.csv")
    runtime = read_csv(paths["results"] / "explainer_runtime.csv")
    ig_sensor = read_csv(
        explanation_dir / "integrated_gradients" / "integrated_gradients_sensor_importance.csv"
    )
    if not sensor.empty:
        st.plotly_chart(
            plots.sensor_importance_bar(sensor),
            width="stretch",
        )
    if not temporal.empty:
        st.plotly_chart(
            plots.temporal_importance_bar(temporal),
            width="stretch",
        )
    if not ig_sensor.empty and "mean_abs_attribution" in ig_sensor:
        st.subheader("Integrated Gradients")
        st.dataframe(ig_sensor.head(12), width="stretch")
    if not runtime.empty:
        st.subheader("Runtime")
        st.plotly_chart(
            plots.explainer_runtime_bar(runtime),
            width="stretch",
        )


def counterfactuals(paths: dict[str, Path]) -> None:
    nearest = read_csv(paths["results"] / "explanations" / "counterfactuals.csv")
    dice = read_csv(
        paths["results"] / "explanations" / "package_backed" / "dice_counterfactuals.csv"
    )
    if not nearest.empty:
        st.subheader("Nearest healthier windows")
        st.dataframe(nearest, width="stretch")
    if not dice.empty:
        st.subheader("DiCE")
        st.dataframe(dice, width="stretch")


def maintenance_advisor(paths: dict[str, Path]) -> None:
    advisor = read_csv(paths["results"] / "maintenance_advisor.csv")
    if advisor.empty:
        st.warning("Maintenance advisor output not found.")
        return
    st.dataframe(advisor, width="stretch")
    st.plotly_chart(
        plots.risk_distribution(advisor),
        width="stretch",
    )


def report_viewer(paths: dict[str, Path]) -> None:
    metadata = read_json(paths["processed"] / "metadata.json")
    if metadata:
        st.json(metadata)
    if paths["report"].exists():
        st.markdown(paths["report"].read_text(encoding="utf-8"))
    else:
        st.warning("Report not found.")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def best_model_row(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty or "test_rmse" not in metrics:
        return {}
    row = metrics.sort_values("test_rmse").iloc[0]
    return row.to_dict()


def format_number(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def display_model_name(value: Any) -> str:
    names = {
        "transformer_sequence": "Transformer",
        "hist_gradient_boosting": "HGB",
        "lstm_sequence": "LSTM",
        "gru_sequence": "GRU",
        "random_forest": "Forest",
        "mlp_neural_regressor": "MLP",
        "ridge": "Ridge",
    }
    return names.get(str(value), str(value))


if __name__ == "__main__":
    main()
