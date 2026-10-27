# XAI evaluation report

Subset: FD001
Window length: 30
Best model: transformer_sequence

## Predictive benchmark

| model | train_windows | validation_rmse | validation_mae | validation_r2 | validation_nasa_score | test_rmse | test_mae | test_r2 | test_nasa_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformer_sequence | 3000 | 15.9426 | 11.5617 | 0.8517 | 23801.9462 | 15.5926 | 10.8793 | 0.7249 | 64137.4521 |
| hist_gradient_boosting | 3000 | 16.5598 | 12.2148 | 0.8400 | 25198.3280 | 16.5204 | 12.5496 | 0.6911 | 70295.2065 |
| lstm_sequence | 3000 | 17.4114 | 13.1350 | 0.8232 | 27668.8032 | 16.9728 | 13.2786 | 0.6740 | 94874.7468 |
| gru_sequence | 3000 | 17.5746 | 12.7812 | 0.8198 | 34966.7196 | 18.1823 | 13.9570 | 0.6259 | 137124.3450 |
| random_forest | 3000 | 18.2814 | 13.7726 | 0.8050 | 36621.9136 | 17.8501 | 13.8506 | 0.6394 | 110562.2239 |
| ridge | 3000 | 19.3140 | 15.7370 | 0.7824 | 25887.6029 | 17.6422 | 14.2225 | 0.6478 | 57505.7516 |
| mlp_neural_regressor | 3000 | 20.5728 | 15.1120 | 0.7531 | 39050.5437 | 24.0014 | 18.7017 | 0.3481 | 154796.5635 |

## Explanation quality

| Metric | Value |
| --- | ---: |
| faithfulness_deletion_delta | 12.4295 |
| stability_neighbor_cosine | 0.4459 |
| consistency_occlusion_lime_spearman | 0.8310 |
| robustness_noise_cosine | 0.9955 |
| sparsity_top3_share | 0.4446 |
| localization_last20pct_share | 0.5079 |

## Trustworthy behavior checks

| Metric | Value |
| --- | ---: |
| subgroup_fairness_mae_gap | 14.4709 |
| subgroup_fairness_max_group_mae | 18.6058 |
| confidence_uncertainty_error_spearman | 0.1501 |
| model_drift_wasserstein_mean | 0.2075 |
| adversarial_top_sensor_prediction_shift | 2.1373 |
| spurious_low_variance_attribution_share | 0.0000 |
| explanation_disagreement_one_minus_top5_jaccard | 0.4628 |

## XAI evaluation framework

| dimension | question | project_impact |
| --- | --- | --- |
| Predictive performance | Is the AI accurate? | Best model transformer_sequence test RMSE 15.593. |
| Interpretability | Can we understand why it predicted? | Global sensor, local sensor, temporal, and counterfactual explanations generated. |
| Faithfulness | Does explanation represent model behavior? | Deletion delta 12.430. |
| Stability | Do similar inputs produce similar explanations? | Neighbor cosine 0.446. |
| Robustness | Does explanation survive perturbations? | Noise cosine 0.995. |
| Fairness | Are decisions consistent across groups? | RUL subgroup MAE gap 14.471. |
| Confidence | Does reliability match confidence? | Uncertainty-error Spearman 0.150. |
| Actionability | Can explanation suggest what to do? | 24 maintenance-advisor rows generated. |
| Human usefulness | Does explanation improve decisions? | Engineer can inspect top sensors and high-risk engines first. |
| Domain impact | Does XAI produce practical benefit? | Maintenance prioritization combines RUL, risk, attribution, and counterfactual evidence. |

## Run scope

Deep sequence models enabled: True. Package-backed SHAP/LIME/DiCE enabled: True.
