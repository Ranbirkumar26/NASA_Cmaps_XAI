from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "cmapss_xai_step_by_step_process.pdf"

INK = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#475569")
TEAL = colors.HexColor("#0F766E")
INDIGO = colors.HexColor("#4338CA")
AMBER = colors.HexColor("#B45309")
LIGHT = colors.HexColor("#F8FAFC")
LINE = colors.HexColor("#CBD5E1")
SOFT_TEAL = colors.HexColor("#CCFBF1")
SOFT_INDIGO = colors.HexColor("#E0E7FF")
SOFT_AMBER = colors.HexColor("#FEF3C7")
SOFT_ROSE = colors.HexColor("#FFE4E6")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, float]:
    return json.loads(path.read_text())


def fmt(value: str | float, digits: int = 3) -> str:
    number = float(value)
    return f"{number:.{digits}f}"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(
    rows: list[list[str | Paragraph]],
    widths: list[float],
    header: bool = True,
    font_size: int = 8,
) -> Table:
    t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SOFT_TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2 == 0:
            style.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F1F5F9")))
    t.setStyle(TableStyle(style))
    return t


def metric_strip(items: list[tuple[str, str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for label, value, detail in items:
        rows.append(
            [
                p(f"<b>{label}</b><br/><font size='16'>{value}</font><br/><font color='#475569'>{detail}</font>", styles["metric"])
            ]
        )
    t = Table([rows], colWidths=[1.75 * inch] * len(items), hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("BACKGROUND", (0, 0), (0, 0), SOFT_INDIGO),
                ("BACKGROUND", (1, 0), (1, 0), SOFT_TEAL),
                ("BACKGROUND", (2, 0), (2, 0), SOFT_AMBER),
                ("BACKGROUND", (3, 0), (3, 0), SOFT_ROSE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def image_block(path: Path, caption: str, width: float = 6.7 * inch) -> list:
    if not path.exists():
        return []
    img = Image(str(path))
    scale = min(width / img.imageWidth, 3.2 * inch / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    return [img, Spacer(1, 0.06 * inch), Paragraph(caption, caption_style())]


def caption_style() -> ParagraphStyle:
    return ParagraphStyle(
        "Caption",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceAfter=6,
    )


def add_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.3)
    canvas.line(doc.leftMargin, 0.52 * inch, A4[0] - doc.rightMargin, 0.52 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.34 * inch, "C-MAPSS XAI Process Guide")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.34 * inch, f"Page {doc.page}")
    canvas.restoreState()


def para_cells(values: Iterable[str], style: ParagraphStyle) -> list[Paragraph]:
    return [p(value, style) for value in values]


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            textColor=INK,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=MUTED,
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=INK,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.4,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=INK,
        ),
        "metric": ParagraphStyle(
            "Metric",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=12,
            textColor=INK,
        ),
        "note": ParagraphStyle(
            "Note",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=MUTED,
            borderColor=LINE,
            borderWidth=0.4,
            borderPadding=7,
            backColor=LIGHT,
            spaceBefore=6,
            spaceAfter=8,
        ),
    }


def build_pdf() -> None:
    styles = build_styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.7 * inch,
        title="C-MAPSS XAI Step-by-Step Process Guide",
        author="Ranbir Kumar and Kumar Manthan",
    )

    cross = read_csv(ROOT / "results" / "xai" / "cross_subset_benchmark.csv")
    fd001_models = read_csv(ROOT / "results" / "xai" / "FD001" / "model_benchmark.csv")
    quality = read_json(ROOT / "results" / "xai" / "FD001" / "xai_quality.json")
    trust = read_json(ROOT / "results" / "xai" / "FD001" / "trustworthiness.json")
    runtime = read_csv(ROOT / "results" / "xai" / "FD001" / "explainer_runtime.csv")
    subgroup = read_csv(ROOT / "results" / "xai" / "FD001" / "subgroup_metrics.csv")

    best_fd001 = fd001_models[0]
    metrics = [
        ("Best FD001 model", best_fd001["model"].replace("_", " "), f"RMSE {fmt(best_fd001['test_rmse'])}"),
        ("High-risk AP", "0.873", "RUL <= 30 ranking"),
        ("Subsets complete", "FD001-FD004", "Prepared and benchmarked"),
        ("Paper package", "12 pages", "16 figures, 14 tables"),
    ]

    story: list = []
    story.append(p("C-MAPSS XAI Project Process Guide", styles["title"]))
    story.append(
        p(
            "Step 0 to final review: what we are doing, how we are doing it, and why each part matters.",
            styles["subtitle"],
        )
    )
    story.append(metric_strip(metrics, styles))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        p(
            "Project team: Ranbir Kumar and Kumar Manthan. Guide: Dr. Poornima S. Project domain: explainable predictive maintenance for aircraft turbofan engines using the NASA C-MAPSS simulated run-to-failure dataset.",
            styles["body"],
        )
    )
    story.append(
        p(
            "Core idea: predict remaining useful life, explain which sensors and time windows drove the prediction, evaluate whether explanations are trustworthy, and show a maintenance engineer how to act on the evidence.",
            styles["note"],
        )
    )

    story.append(p("Executive view", styles["h1"]))
    story.append(
        p(
            "The project treats C-MAPSS as a research pipeline, not as a single notebook. Raw NASA files are parsed into versioned Parquet data, transformed into train-only-normalized time windows, validated for leakage risks, used for model benchmarking, explained through multiple XAI methods, measured through explanation-quality metrics, and displayed through a Streamlit decision-support dashboard.",
            styles["body"],
        )
    )
    story.extend(image_block(ROOT / "paper" / "figures" / "proposed_architecture.png", "Figure 1. Current project architecture from raw NASA files to model, XAI, dashboard, and report outputs."))

    story.append(PageBreak())
    story.append(p("Step-by-step plan", styles["h1"]))
    steps = [
        (
            "Step 0: Frame the problem",
            "We define the project as explainable RUL prediction for turbofan maintenance.",
            "We use NASA C-MAPSS, model RUL from multivariate sensor windows, and evaluate both prediction and explanation behavior.",
            "A final review needs a clear safety-oriented question, not only a regression score.",
            "Problem statement, evaluation checklist, final-review story.",
        ),
        (
            "Step 1: Acquire and scope data",
            "We use NASA CMAPSSData.zip and support FD001 to FD004.",
            "The parser reads train, test, and RUL files, then assigns official NASA columns: unit, cycle, three settings, and 21 sensors.",
            "C-MAPSS gives controlled run-to-failure trajectories with known operating-condition and fault-mode variation.",
            "Raw files under data/raw and subset metadata under data/processed.",
        ),
        (
            "Step 2: Create RUL targets",
            "We compute target remaining useful life for every train cycle and align test windows with the official test RUL file.",
            "Train RUL uses max_cycle minus current cycle, capped where configured. Test RUL adds official final RUL to observed test sequence length.",
            "RUL target construction is the foundation for fair model comparison.",
            "Processed train.parquet, test.parquet, test_rul.parquet.",
        ),
        (
            "Step 3: Normalize safely",
            "We scale sensor and setting features with train data only.",
            "Scaler parameters are saved as normalization.json and scaler.pkl, then reused for validation and test artifacts.",
            "Train-only normalization prevents test leakage and keeps XAI inputs consistent with model inputs.",
            "normalization.json, scaler.pkl, selected feature list.",
        ),
        (
            "Step 4: Build time windows",
            "We convert trajectories into fixed-length sequence windows for ML and deep models.",
            "Each window stores unit id, start cycle, end cycle, target RUL, and feature matrix. Metadata maps each row back to the engine and cycle range.",
            "Window metadata is essential for debugging explanations and showing local examples in the dashboard.",
            "windows.npz and window_metadata.parquet.",
        ),
        (
            "Step 5: Validate data",
            "We run schema, cycle, dtype, NaN, Inf, duplicate, and window-boundary checks.",
            "Validation confirms cycle starts, monotonic cycles per engine, no cross-unit windows, and expected train/test unit counts.",
            "This catches hidden preprocessing errors before model training creates misleading results.",
            "validation_report.txt for each subset.",
        ),
        (
            "Step 6: Generate EDA and reports",
            "We profile sensors, engine trajectories, correlations, operating conditions, and baseline feature importance.",
            "The pipeline writes sensor reports, statistics figures, engine profile figures, and a dataset card.",
            "EDA explains the data before modeling and gives paper-ready figures.",
            "reports/figures, sensor_report.csv, dataset_card.md.",
        ),
        (
            "Step 7: Benchmark predictive models",
            "We compare classical ML and deep sequence models.",
            "Models include Ridge, Random Forest, HistGradientBoosting, MLPRegressor, LSTM, GRU, and Transformer sequence regressors.",
            "Predictive performance answers whether the AI is accurate enough to support explanations.",
            "model_benchmark.csv, test_predictions.csv, trained model artifacts.",
        ),
        (
            "Step 8: Generate explanations",
            "We compute global, local, temporal, and counterfactual explanations.",
            "Methods include SHAP, LIME, DiCE, sensor occlusion, temporal occlusion, and Integrated Gradients.",
            "Multiple explainers reduce dependence on one method and expose disagreement between explanation styles.",
            "global importance CSVs, local examples, counterfactuals, temporal attribution.",
        ),
        (
            "Step 9: Evaluate explanation quality",
            "We measure explanations instead of only showing plots.",
            "The framework reports faithfulness, stability, consistency, robustness, sparsity, localization, and runtime cost.",
            "XAI projects need evidence that explanations reflect model behavior and remain useful under noise.",
            "xai_quality.json and explainer_runtime.csv.",
        ),
        (
            "Step 10: Investigate trust behavior",
            "We test subgroup gaps, uncertainty, drift, spurious feature share, adversarial shift, and explanation disagreement.",
            "Prediction intervals, subgroup metrics, and trustworthiness metrics are computed from saved predictions and explanations.",
            "Trust checks help us state both strengths and limits honestly.",
            "prediction_intervals.csv, subgroup_metrics.csv, trustworthiness.json.",
        ),
        (
            "Step 11: Build decision support",
            "We convert prediction plus explanation into a maintenance-style advisory view.",
            "Dashboard pages show engine explorer, sensor explorer, operating conditions, model benchmark, XAI explanations, counterfactuals, advisor, and report viewer.",
            "This shows practical impact: the engineer can prioritize an engine and inspect supporting sensors.",
            "Streamlit dashboard and maintenance_advisor.csv.",
        ),
        (
            "Step 12: Prepare final review package",
            "We align report, PPT, dashboard, and demo flow around the same evidence.",
            "The paper explains research detail, the PPT tells the review story, and the dashboard proves the project runs from cached artifacts.",
            "A final review needs a complete, coherent delivery rather than disconnected files.",
            "IEEE-style PDF report, detailed PPT, dashboard, and demo script.",
        ),
    ]

    step_rows = [[p("<b>Step</b>", styles["small"]), p("<b>What</b>", styles["small"]), p("<b>How</b>", styles["small"]), p("<b>Why</b>", styles["small"]), p("<b>Output</b>", styles["small"])]]
    for name, what, how, why, output in steps:
        step_rows.append(para_cells([f"<b>{name}</b>", what, how, why, output], styles["small"]))
    story.append(table(step_rows, [1.2 * inch, 1.45 * inch, 2.05 * inch, 1.75 * inch, 1.4 * inch], font_size=7))

    story.append(p("Dataset and preprocessing detail", styles["h1"]))
    dataset_rows = [
        ["Subset", "Train units", "Test units", "Operating conditions", "Fault modes"],
        ["FD001", "100", "100", "1", "1"],
        ["FD002", "260", "259", "6", "1"],
        ["FD003", "100", "100", "1", "2"],
        ["FD004", "249", "248", "6", "2"],
    ]
    story.append(table(dataset_rows, [1.1 * inch, 1.2 * inch, 1.2 * inch, 1.6 * inch, 1.2 * inch], font_size=8))
    story.append(
        p(
            "FD001 remains the main explained case because it has one operating condition and one fault mode. FD002 to FD004 are included to show that the parser, validation, model runner, and XAI outputs expand across harder multi-condition and multi-fault subsets.",
            styles["body"],
        )
    )
    story.extend(image_block(ROOT / "paper" / "figures" / "fd001_engine_profile.png", "Figure 2. Example FD001 engine profile: normalized sensor trends with RUL curve."))

    story.append(p("Model benchmark", styles["h1"]))
    model_rows = [[p("<b>Model</b>", styles["small"]), p("<b>Validation RMSE</b>", styles["small"]), p("<b>Test RMSE</b>", styles["small"]), p("<b>Test MAE</b>", styles["small"]), p("<b>Test R2</b>", styles["small"])]]
    for row in fd001_models[:7]:
        model_rows.append(
            para_cells(
                [
                    row["model"].replace("_", " "),
                    fmt(row["validation_rmse"]),
                    fmt(row["test_rmse"]),
                    fmt(row["test_mae"]),
                    fmt(row["test_r2"]),
                ],
                styles["small"],
            )
        )
    story.append(table(model_rows, [2.25 * inch, 1.2 * inch, 1 * inch, 1 * inch, 0.9 * inch], font_size=8))
    story.extend(image_block(ROOT / "paper" / "figures" / "model_test_rmse.png", "Figure 3. FD001 model test RMSE comparison."))

    story.append(p("Cross-subset benchmark", styles["h1"]))
    cross_rows = [[p("<b>Subset</b>", styles["small"]), p("<b>Best model</b>", styles["small"]), p("<b>Test RMSE</b>", styles["small"]), p("<b>Test MAE</b>", styles["small"]), p("<b>Interval coverage</b>", styles["small"]), p("<b>Successful explainers</b>", styles["small"])]]
    for row in cross:
        cross_rows.append(
            para_cells(
                [
                    row["subset"],
                    row["model"].replace("_", " "),
                    fmt(row["test_rmse"]),
                    fmt(row["test_mae"]),
                    fmt(row["interval_coverage"]),
                    row["successful_explainers"],
                ],
                styles["small"],
            )
        )
    story.append(table(cross_rows, [0.75 * inch, 2.25 * inch, 1 * inch, 0.95 * inch, 1.2 * inch, 1.25 * inch], font_size=8))
    story.append(
        p(
            "Interpretation: FD001 and FD003 perform best in current runs. FD002 and FD004 are harder because they include six operating conditions, and the small default runs favor classical baselines. This is why final claims focus on validated pipeline completeness plus FD001 explanation depth.",
            styles["note"],
        )
    )

    story.append(p("XAI generation and evaluation", styles["h1"]))
    story.append(
        p(
            "The XAI layer answers which sensors mattered, which time points mattered, whether a local engine case has similar explanation behavior, and how much feature change would move the output toward a safer RUL. Explanation methods are matched to time-series tabular data: SHAP and LIME for feature attribution, Integrated Gradients for deep sequence attribution, DiCE for counterfactuals, and occlusion for model-agnostic sensor and temporal ablation.",
            styles["body"],
        )
    )
    quality_rows = [
        ["Metric", "FD001 value", "Meaning"],
        ["Consistency", fmt(quality["consistency_occlusion_lime_spearman"]), "Agreement between occlusion and LIME ranks"],
        ["Faithfulness", fmt(quality["faithfulness_deletion_delta"]), "Prediction change after removing important evidence"],
        ["Localization", fmt(quality["localization_last20pct_share"]), "Share of attribution in recent window positions"],
        ["Robustness", fmt(quality["robustness_noise_cosine"]), "Attribution similarity after noise"],
        ["Sparsity", fmt(quality["sparsity_top3_share"]), "How concentrated top sensors are"],
        ["Stability", fmt(quality["stability_neighbor_cosine"]), "Similarity for nearby examples"],
    ]
    story.append(table(quality_rows, [1.35 * inch, 1.1 * inch, 4.45 * inch], font_size=8))
    story.extend(image_block(ROOT / "paper" / "figures" / "top_sensor_importance.png", "Figure 4. Top FD001 sensor attributions across explainers."))

    story.append(p("Runtime, trust, and subgroup behavior", styles["h1"]))
    runtime_rows = [[p("<b>Explainer</b>", styles["small"]), p("<b>Seconds per sample</b>", styles["small"]), p("<b>Peak memory MB</b>", styles["small"]), p("<b>Status</b>", styles["small"])]]
    for row in runtime:
        runtime_rows.append(
            para_cells(
                [
                    row["explainer"].replace("_", " "),
                    fmt(row["seconds_per_sample"], 4),
                    fmt(row["peak_memory_mb"]),
                    row["status"],
                ],
                styles["small"],
            )
        )
    story.append(table(runtime_rows, [2.4 * inch, 1.35 * inch, 1.3 * inch, 1 * inch], font_size=8))
    story.append(
        p(
            "Runtime matters because a dashboard or review demo cannot depend only on slow explainers. Fast occlusion gives quick evidence, while SHAP, LIME, DiCE, and Integrated Gradients provide deeper validation on smaller samples.",
            styles["body"],
        )
    )
    trust_rows = [
        ["Trust dimension", "FD001 value", "Interpretation"],
        ["Adversarial sensor shift", fmt(trust["adversarial_top_sensor_prediction_shift"]), "Prediction sensitivity to top sensor perturbation"],
        ["Confidence-error Spearman", fmt(trust["confidence_uncertainty_error_spearman"]), "Association between uncertainty and error"],
        ["Explanation disagreement", fmt(trust["explanation_disagreement_one_minus_top5_jaccard"]), "Top-feature disagreement across explainers"],
        ["Model drift", fmt(trust["model_drift_wasserstein_mean"]), "Mean distribution shift indicator"],
        ["Spurious low-variance share", fmt(trust["spurious_low_variance_attribution_share"]), "Attribution assigned to nearly static sensors"],
        ["Subgroup MAE gap", fmt(trust["subgroup_fairness_mae_gap"]), "Gap across high, medium, low RUL groups"],
    ]
    story.append(table(trust_rows, [1.8 * inch, 1.1 * inch, 4.25 * inch], font_size=8))
    story.extend(image_block(ROOT / "paper" / "figures" / "subgroup_mae.png", "Figure 5. FD001 subgroup MAE by risk group."))

    story.append(PageBreak())
    story.append(p("Decision-support value", styles["h1"]))
    story.append(
        p(
            "The decision-support layer translates RUL plus XAI into maintenance language. Instead of saying only that a model predicts 18 cycles, the dashboard shows the risk class, top contributing sensors, confidence or uncertainty, local explanation examples, and counterfactual evidence. This supports the teacher requirement that XAI should improve human decision-making.",
            styles["body"],
        )
    )
    story.extend(image_block(ROOT / "paper" / "figures" / "high_risk_precision_recall.png", "Figure 6. FD001 high-risk precision-recall curve. Average precision is 0.873."))
    story.extend(image_block(ROOT / "paper" / "figures" / "dashboard_overview.png", "Figure 7. Streamlit final-review dashboard overview."))

    story.append(PageBreak())
    story.append(p("How final review should be presented", styles["h1"]))
    review_rows = [
        ["Review moment", "What to show", "Why it matters"],
        ["Opening", "Problem, dataset, and objective", "Frames the project as maintenance decision support"],
        ["Technical core", "Pipeline, preprocessing, validation, and model benchmark", "Shows engineering rigor and no leakage"],
        ["XAI core", "Sensor attribution, temporal attribution, counterfactuals", "Shows model behavior from multiple angles"],
        ["Quality checks", "Faithfulness, stability, robustness, runtime, subgroup gap", "Shows evaluation beyond screenshots"],
        ["Live demo", "Dashboard overview, XAI explanations, maintenance advisor", "Shows working decision-support artifact"],
        ["Limitations", "Expert feedback pending, uncalibrated intervals, larger tuning deferred", "Keeps claims honest and research-like"],
    ]
    story.append(table(review_rows, [1.3 * inch, 3.0 * inch, 2.55 * inch], font_size=8))

    story.append(PageBreak())
    story.append(p("Final outputs and file map", styles["h1"]))
    file_rows = [
        ["Output", "Location", "Purpose"],
        ["Processed data", "data/processed/FD001 to FD004", "Parquet tables, windows, metadata, scaler, validation"],
        ["Model and XAI results", "results/xai/<subset>", "Predictions, benchmarks, explanations, runtime, trust checks"],
        ["Report figures", "paper/figures", "Architecture, model, XAI, quality, uncertainty, dashboard images"],
        ["IEEE-style paper", "paper/ieee_report.pdf", "Formal write-up for submission"],
        ["PPT deck", "presentation/cmapss_xai_proposal.pptx", "Final review presentation"],
        ["Dashboard", "src/cmapss_xai/dashboard/app.py", "Interactive final demo using cached artifacts"],
        ["Expert packet", "paper/expert_review_packet.md", "Protocol for real human expert feedback"],
    ]
    story.append(table(file_rows, [1.45 * inch, 2.5 * inch, 3.0 * inch], font_size=8))

    story.append(p("Current limitations", styles["h1"]))
    limitations = [
        "Human expert feedback is prepared through a packet and template, but real expert responses are not fabricated.",
        "Prediction intervals are ensemble-based and uncalibrated, so coverage is reported as evidence rather than claimed as final uncertainty calibration.",
        "FD002 and FD004 are harder multi-condition subsets; larger tuning and condition-aware normalization are future work.",
        "Deep model tuning is intentionally limited to keep final review stable and reproducible.",
        "The dashboard reads cached results and does not train models live, which makes the demo reliable.",
    ]
    story.append(table([[p("<b>Limitation</b>", styles["small"])]] + [[p(item, styles["small"])] for item in limitations], [6.7 * inch], font_size=8))

    story.append(p("Final takeaway", styles["h1"]))
    story.append(
        p(
            "This project starts with a clean predictive-maintenance problem, builds a reproducible C-MAPSS data pipeline, benchmarks suitable ML and sequence models, generates multiple explanation types, evaluates explanation quality and trust behavior, and ends with a dashboard and review package that explain why a maintenance engineer should trust or question a prediction.",
            styles["note"],
        )
    )

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)


if __name__ == "__main__":
    build_pdf()
    print(OUT)
