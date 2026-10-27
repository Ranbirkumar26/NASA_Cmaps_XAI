# PPT Generation Prompt

Create a professional investor-style PowerPoint deck under 10 slides for this project:

Project title: Explainable Predictive Maintenance for Aircraft Turbofan Engines Using NASA C-MAPSS.

Audience: academic evaluators, technical reviewers, and potential research sponsors. Tone must be clear, technical, and convincing. Show novelty: this is not only RUL prediction; it is a full explainable maintenance decision-support framework.

Use these slide contents:

1. Title slide
   - Title: Explainable predictive maintenance for aircraft turbofan engines.
   - Subtitle: RUL prediction with sensor, temporal, counterfactual, uncertainty, and trust evidence.
   - Key metrics: Transformer, FD001 test RMSE 15.593, FD001 test MAE 10.879, FD001-FD004 support.

2. Problem statement
   - Aircraft engines need early failure prediction.
   - Black-box RUL prediction is not enough for maintenance decisions.
   - Engineers need to know which sensors caused the warning, whether explanation is reliable, and what action to prioritize.

3. Dataset
   - NASA C-MAPSS Jet Engine Simulated Data.
   - Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data.
   - FD001: 100 train engines, 100 test engines, 1 operating condition, 1 fault mode.
   - FD002-FD004 add multiple conditions and fault modes.
   - Each row has engine id, cycle, 3 settings, 21 sensors.

4. Proposed architecture
   - NASA raw data to validation.
   - Preprocessing to Parquet and window arrays.
   - Model benchmark.
   - XAI layer.
   - Explanation-quality evaluation.
   - Streamlit dashboard and maintenance advisor.

5. Models and XAI methods
   - Models: Ridge, Random Forest, HistGradientBoosting, MLPRegressor, LSTM, GRU, Transformer.
   - XAI: SHAP, LIME, DiCE, sensor occlusion, temporal occlusion, Integrated Gradients.
   - Explain why this is multimethod and robust.

6. Predictive results
   - Show FD001 RMSE chart.
   - Transformer test RMSE 15.593.
   - HistGradientBoosting test RMSE 16.520.
   - LSTM test RMSE 16.973.
   - State that Transformer currently leads.

7. XAI results
   - Top sensors from occlusion: sensor_11, sensor_14, sensor_4, sensor_8, sensor_13.
   - Integrated Gradients agrees on sensor_14, sensor_4, sensor_13, sensor_11.
   - Recent cycles dominate temporal attribution.
   - DiCE gives counterfactual healthier sensor patterns.

8. Trust, failure cases, and evaluation
   - Faithfulness deletion delta 12.430.
   - Robustness noise cosine 0.995.
   - Interval coverage 0.668, needs calibration.
   - Subgroup MAE gap 14.471, medium-risk group needs improvement.
   - Runtime: occlusion and IG are fast; SHAP and DiCE should run offline.

9. Novelty and future plan
   - Novelty: reproducible C-MAPSS XAI pipeline, full explanation evaluation, runtime metrics, uncertainty, subgroup fairness, maintenance-advisor output, expert feedback protocol.
   - Future: matched-budget FD001-FD004 runs, conformal intervals, fairness mitigation, expert feedback, final IEEE paper, live dashboard demo.

Design requirements:

- Use clean 16:9 layout.
- Use editable native charts, not screenshots, when showing metrics.
- Keep under 10 slides.
- Do not fabricate expert feedback.
- Mention that expert feedback is pending until real reviewer ratings are collected.
