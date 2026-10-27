# XAI evaluation report

Subset: FD003
Window length: 30
Best model: hist_gradient_boosting

## Predictive benchmark

| model | train_windows | validation_rmse | validation_mae | validation_r2 | validation_nasa_score | test_rmse | test_mae | test_r2 | test_nasa_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | 1000 | 15.4293 | 11.5801 | 0.8632 | 23286.1696 | 14.7881 | 10.5300 | 0.6906 | 78499.9043 |
| random_forest | 1000 | 16.2493 | 12.5045 | 0.8483 | 27226.7570 | 16.0145 | 11.7673 | 0.6372 | 97997.7588 |
| mlp_neural_regressor | 1000 | 16.9633 | 13.4314 | 0.8347 | 24200.9026 | 16.6241 | 12.8481 | 0.6090 | 79228.6418 |
| lstm_sequence | 1000 | 17.0349 | 12.8320 | 0.8333 | 29283.4356 | 17.9083 | 11.9237 | 0.5463 | 124283.9385 |
| ridge | 1000 | 22.3926 | 17.8169 | 0.7119 | 85363.1058 | 22.3087 | 17.7951 | 0.2959 | 191355.3182 |
| transformer_sequence | 1000 | 24.4950 | 18.0627 | 0.6553 | 127434.5859 | 18.2487 | 12.0217 | 0.5289 | 241615.8793 |
| gru_sequence | 1000 | 26.4073 | 19.4719 | 0.5993 | 247462.7127 | 19.6552 | 13.7870 | 0.4534 | 228594.1418 |

## Explanation quality

| Metric | Value |
| --- | ---: |
| faithfulness_deletion_delta | 23.4316 |
| stability_neighbor_cosine | 0.6198 |
| consistency_occlusion_lime_spearman | 0.8522 |
| robustness_noise_cosine | 0.8771 |
| sparsity_top3_share | 0.6300 |
| localization_last20pct_share | 0.6424 |

## Trustworthy behavior checks

| Metric | Value |
| --- | ---: |
| subgroup_fairness_mae_gap | 11.2088 |
| subgroup_fairness_max_group_mae | 20.2105 |
| confidence_uncertainty_error_spearman | 0.1994 |
| model_drift_wasserstein_mean | 0.1765 |
| adversarial_top_sensor_prediction_shift | 2.6535 |
| spurious_low_variance_attribution_share | 0.0000 |
| explanation_disagreement_one_minus_top5_jaccard | 0.5268 |

## XAI evaluation framework

| dimension | question | project_impact |
| --- | --- | --- |
| Predictive performance | Is the AI accurate? | Best model hist_gradient_boosting test RMSE 14.788. |
| Interpretability | Can we understand why it predicted? | Global sensor, local sensor, temporal, and counterfactual explanations generated. |
| Faithfulness | Does explanation represent model behavior? | Deletion delta 23.432. |
| Stability | Do similar inputs produce similar explanations? | Neighbor cosine 0.620. |
| Robustness | Does explanation survive perturbations? | Noise cosine 0.877. |
| Fairness | Are decisions consistent across groups? | RUL subgroup MAE gap 11.209. |
| Confidence | Does reliability match confidence? | Uncertainty-error Spearman 0.199. |
| Actionability | Can explanation suggest what to do? | 8 maintenance-advisor rows generated. |
| Human usefulness | Does explanation improve decisions? | Engineer can inspect top sensors and high-risk engines first. |
| Domain impact | Does XAI produce practical benefit? | Maintenance prioritization combines RUL, risk, attribution, and counterfactual evidence. |

## Run scope

Deep sequence models enabled: True. Package-backed SHAP/LIME/DiCE enabled: True.
