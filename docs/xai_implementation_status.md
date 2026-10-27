# XAI implementation status

## Implemented

- Predictive benchmarking with Ridge, Random Forest, HistGradientBoosting, MLP neural regressor, LSTM, GRU, and Transformer.
- FD001 to FD004 time-series window training with unit-level validation split.
- Model metrics: RMSE, MAE, R2, and NASA score.
- Time-series XAI:
  - Package-backed SHAP attribution.
  - Package-backed LIME attribution.
  - Package-backed DiCE counterfactuals.
  - Captum Integrated Gradients for torch sequence models.
  - SHAP-style sensor occlusion attribution.
  - Temporal occlusion attribution.
  - LIME-style local surrogate attribution.
  - DiCE-style nearest healthier-window counterfactuals.
- Explanation quality metrics:
  - Faithfulness by deletion delta.
  - Stability by nearest-neighbor attribution cosine.
  - Consistency by occlusion-vs-surrogate Spearman correlation.
  - Robustness by noise attribution cosine.
  - Sparsity by top-3 attribution share.
  - Localization by last-20-percent temporal attribution share.
- Trust checks:
  - Risk-subgroup MAE gap.
  - Ensemble uncertainty vs error correlation.
  - Train-test drift by Wasserstein distance.
  - Adversarial top-sensor prediction shift.
  - Low-variance sensor attribution share.
  - Explanation disagreement by top-5 Jaccard distance.
- Decision-support output:
  - Maintenance-advisor CSV with risk level, top sensors, and inspection window.
- Runtime and memory benchmarking:
  - `results/xai/<subset>/explainer_runtime.csv`.
- Uncertainty and subgroup outputs:
  - `prediction_intervals.csv`.
  - `subgroup_metrics.csv`.
  - `fairness_mitigation.csv` with original vs balanced-sampling comparison.
- Dashboard:
  - Streamlit artifact viewer with engine, sensor, operating-condition, model, XAI, counterfactual, maintenance, and report pages.
- Documentation artifacts:
  - `paper/xai_report.md`.
  - `paper/ieee_report.tex`.
  - `presentation/cmapss_xai_proposal.pptx`.
  - `paper/expert_review_packet.md`.
  - `paper/expert_feedback_template.csv`.
- XAI evaluation framework table mapping project results to required dimensions.

## Run

```bash
.venv311/bin/cmapss-xai run-xai --subset FD001 --window-length 30 --include-deep-models --deep-epochs 12 --exact-xai --integrated-gradients --benchmark-explainers --prediction-intervals --tuning-grid
```

## Scope note

Deep sequence models and package-backed XAI run in `.venv311` because the local `.venv` uses Python 3.14. Real human expert feedback remains pending until reviewer responses are collected. FD002 to FD004 currently have smoke-scale XAI benchmarks; final paper-grade cross-subset claims should use matched training budgets.
