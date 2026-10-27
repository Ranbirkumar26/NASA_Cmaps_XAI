# Project Proposal Brief

## Title

Explainable Predictive Maintenance for Aircraft Turbofan Engines Using NASA C-MAPSS

## Problem Statement

Aircraft engine maintenance teams need early, trustworthy remaining useful life predictions. A normal ML model can predict RUL, but it does not explain which sensor readings caused a low-RUL warning. This project solves that gap by combining RUL prediction with explainable AI so a maintenance engineer can see the predicted risk, important sensors, important timesteps, counterfactual healthier patterns, uncertainty, and subgroup reliability.

## Dataset

Dataset: NASA C-MAPSS Jet Engine Simulated Data  
Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data  
Download: https://data.nasa.gov/docs/legacy/CMAPSSData.zip

The project supports FD001 to FD004. FD001 is the main research subset because it has 100 train engines, 100 test engines, one operating condition, and one high-pressure-compressor degradation fault mode. Each row contains engine id, cycle, 3 operating settings, and 21 sensor readings. The target is remaining useful life, capped at 125 cycles.

## Framework

The framework has four layers:

1. Dataset and validation layer: parse raw NASA files, validate schema and cycles, compute RUL, normalize using train-only scaler, save Parquet tables, windows, metadata, feature registry, and dataset hash.
2. Predictive model layer: benchmark Ridge, Random Forest, HistGradientBoosting, MLPRegressor, LSTM, GRU, and Transformer sequence regressors.
3. XAI layer: generate SHAP, LIME, DiCE, sensor occlusion, temporal occlusion, nearest healthier-window counterfactuals, and Captum Integrated Gradients.
4. Evaluation and decision-support layer: measure faithfulness, stability, consistency, robustness, sparsity, localization, subgroup fairness, uncertainty-error confidence, drift, adversarial sensitivity, runtime, memory, and maintenance actionability.

## Expected Results

Expected technical result: a model that predicts RUL with strong RMSE and MAE while explaining which sensors and timesteps influenced each prediction.

Current FD001 result: Transformer sequence regressor is best with test RMSE 15.593 and test MAE 10.879.

Expected XAI result: engineers can prioritize inspection using high-risk engine rows, top sensors, temporal evidence, counterfactual patterns, and uncertainty. Current explainers show agreement around sensors such as `sensor_11`, `sensor_14`, `sensor_4`, and `sensor_13`, while also revealing limitations such as LIME disagreement and weaker medium-risk performance.
