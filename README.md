# C-MAPSS XAI Predictive Maintenance

Explainable remaining useful life prediction for aircraft turbofan engines using the NASA C-MAPSS dataset. The project trains baseline RUL models, generates model explanations, evaluates explanation quality, and exposes cached results through a Streamlit dashboard.

## Project Scope

This repository focuses on FD001 as the main reported experiment.

Implemented:

- NASA C-MAPSS download, parsing, validation, preprocessing, and fingerprinting.
- Train-only normalization and 30-cycle window generation.
- Classical models: Ridge, Random Forest, HistGradientBoosting, and MLP.
- Sequence models: LSTM, GRU, and Transformer.
- Explainability: sensor occlusion, temporal occlusion, SHAP, LIME, DiCE, and Integrated Gradients.
- XAI evaluation: faithfulness, random deletion control, stability, consistency, robustness, sparsity, temporal localization, runtime, memory, uncertainty, subgroup gap, and explanation disagreement.
- Streamlit dashboard for cached model and XAI outputs.
- IEEE-style report and final-review presentation artifacts.

Deferred or limited:

- Larger multi-seed tuning.
- Full matched-budget FD002 to FD004 model comparison.
- Real human expert study.
- Calibrated conformal prediction intervals.

## Repository Layout

```text
configs/                  Experiment and pipeline configuration
data/                     Local raw and processed NASA data, ignored by git
docs/                     Research notes, outline, and references
experiments/              Experiment metadata and tuning outputs
literature/               Literature notes and papers, optional
notebooks/                Optional exploration notebooks
paper/                    IEEE report source, figures, PDF, and review material
presentation/             Final review deck
reports/                  Generated dataset and statistics reports
results/                  Generated model, XAI, and dashboard artifacts
src/cmapss_xai/           Main Python package
tests/                    Unit and smoke tests
tools/                    Utility scripts
```

## Requirements

Use Python 3.11 for deep models and package-backed XAI.

Core stack:

- pandas
- numpy
- scikit-learn
- scipy
- PyYAML
- joblib
- pyarrow
- plotly
- streamlit
- rich
- tqdm
- pytest

Deep and XAI extras:

- torch
- captum
- shap
- lime
- dice-ml

## Setup

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,deep-xai]"
```

If `python3.11` is unavailable, install Python 3.11 first. PyTorch support may be unstable on newer Python versions.

## Data

NASA C-MAPSS is public:

- NASA Open Data page: `https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data`
- Legacy archive: `https://data.nasa.gov/docs/legacy/CMAPSSData.zip`

Raw and processed data are intentionally ignored by git. Regenerate them locally with the commands below.

## Reproduce FD001 Pipeline

Download and extract data:

```bash
.venv311/bin/cmapss-xai download
```

Prepare FD001:

```bash
.venv311/bin/cmapss-xai prepare --subset FD001
```

Validate FD001:

```bash
.venv311/bin/cmapss-xai validate --subset FD001
```

Generate dataset fingerprint:

```bash
.venv311/bin/cmapss-xai fingerprint --subset FD001
```

Run full FD001 benchmark and XAI pipeline:

```bash
.venv311/bin/cmapss-xai run-xai \
  --subset FD001 \
  --window-length 30 \
  --max-train-windows 3000 \
  --xai-samples 24 \
  --include-deep-models \
  --deep-epochs 4 \
  --deep-batch-size 128 \
  --deep-hidden-size 48 \
  --deep-learning-rate 0.001 \
  --deep-dropout 0.1 \
  --exact-xai \
  --integrated-gradients \
  --benchmark-explainers \
  --prediction-intervals \
  --tuning-grid
```

Expected key outputs:

```text
data/processed/FD001/train.parquet
data/processed/FD001/test.parquet
data/processed/FD001/test_rul.parquet
data/processed/FD001/windows.npz
data/processed/FD001/window_metadata.parquet
data/processed/FD001/metadata.json
data/processed/FD001/feature_registry.yaml
data/processed/FD001/dataset_hash.txt
results/xai/FD001/model_benchmark.csv
results/xai/FD001/test_predictions.csv
results/xai/FD001/xai_quality.json
results/xai/FD001/trustworthiness.json
results/xai/FD001/explainer_runtime.csv
results/xai/FD001/subgroup_metrics.csv
results/xai/FD001/prediction_intervals.csv
reports/xai_evaluation/FD001/xai_evaluation_report.md
```

## Optional Cross-Subset Smoke Run

FD002 to FD004 are supported by the parser and preprocessing pipeline. The paper does not claim matched-budget final results for these subsets.

```bash
.venv311/bin/cmapss-xai run-xai \
  --subsets FD001 FD002 FD003 FD004 \
  --window-length 30 \
  --max-train-windows 1000 \
  --xai-samples 8 \
  --include-deep-models \
  --deep-epochs 3 \
  --exact-xai \
  --integrated-gradients \
  --benchmark-explainers \
  --prediction-intervals
```

## Dashboard

Run dashboard after generating `results/xai/FD001` artifacts:

```bash
.venv311/bin/streamlit run src/cmapss_xai/dashboard/app.py
```

Default local URL:

```text
http://localhost:8501
```

Dashboard pages:

- Engine explorer
- Sensor explorer
- Operating conditions
- Model benchmark
- XAI explanations
- Counterfactuals
- Maintenance advisor
- Report viewer

Dashboard reads existing CSV, JSON, and figure artifacts. It does not train models inside Streamlit.

## Tests

Run unit tests:

```bash
.venv311/bin/pytest
```

Useful checks covered:

- Parser and preprocessing behavior
- Hashing stability
- Window metadata alignment
- Deep model smoke tests
- Integrated Gradients shape checks
- Runtime benchmark schema
- Prediction interval schema
- Subgroup metrics schema

## Report and Presentation

Paper source:

```text
paper/ieee_report.tex
```

Compiled paper:

```text
paper/ieee_report.pdf
```

Presentation:

```text
presentation/cmapss_xai_proposal.pptx
```

LaTeX build from project root:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error ieee_report.tex
```

## Current FD001 Headline Results

All-window FD001 benchmark:

- Transformer RMSE: 15.59 cycles
- Transformer MAE: 10.88 cycles
- Ridge has lower all-window NASA score than Transformer

Standard final-window protocol:

- Transformer capped RMSE: 14.42 cycles
- Transformer uncapped RMSE: 15.31 cycles
- Transformer capped final NASA score: 377.71
- Transformer uncapped final NASA score: 401.00

XAI summary:

- Occlusion, SHAP, and Integrated Gradients agree most clearly on sensors 14, 11, and 13.
- Top-3 sensor deletion shifts predictions by 12.43 cycles.
- Random-3 deletion shifts predictions by 3.49 +/- 0.70 cycles over 50 random draws.
- Robustness under small perturbation is high, but neighbor stability is moderate.

## Submission Notes

Recommended submission contents:

- `src/`
- `configs/`
- `tests/`
- `docs/references.bib`
- `paper/ieee_report.tex`
- `paper/ieee_report.pdf`
- `paper/figures/`
- `presentation/cmapss_xai_proposal.pptx`
- `pyproject.toml`
- `README.md`

Do not submit local runtime folders:

- `.venv/`
- `.venv311/`
- `.ruff_cache/`
- `.pytest_cache/`
- `.chart-data-*`
- `.pptx_build/`
- `tmp/`
- OS files such as `.DS_Store`

Data and generated outputs can be regenerated. Include them only if the evaluator expects an offline demo or supplementary artifacts.

## Troubleshooting

If `torch` install fails:

```bash
python --version
```

Use Python 3.11.

If dashboard opens but tables are empty, run the FD001 XAI command first.

If LaTeX fails, remove stale auxiliary files and rebuild from `paper/`:

```bash
cd paper
latexmk -C
latexmk -pdf -interaction=nonstopmode -halt-on-error ieee_report.tex
```

If NASA download fails, manually download `CMAPSSData.zip` from the NASA legacy archive and place it in:

```text
data/raw/CMAPSSData.zip
```

Then rerun:

```bash
.venv311/bin/cmapss-xai download
```

