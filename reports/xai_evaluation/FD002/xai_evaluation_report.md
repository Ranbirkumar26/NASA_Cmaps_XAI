# XAI evaluation report

Subset: FD002
Window length: 30
Best model: ridge

## Predictive benchmark

| model | train_windows | validation_rmse | validation_mae | validation_r2 | validation_nasa_score | test_rmse | test_mae | test_r2 | test_nasa_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | 1000 | 21.2210 | 17.0694 | 0.7417 | 111574.7753 | 21.8676 | 17.6746 | 0.5022 | 282667.5049 |
| hist_gradient_boosting | 1000 | 22.7543 | 17.9984 | 0.7031 | 196352.6072 | 22.2400 | 17.9238 | 0.4851 | 411127.2448 |
| random_forest | 1000 | 24.1558 | 20.2187 | 0.6654 | 218768.6556 | 22.6816 | 19.4970 | 0.4644 | 380321.0946 |
| mlp_neural_regressor | 1000 | 38.1123 | 32.1824 | 0.1670 | 3230040.4357 | 36.0382 | 30.6322 | -0.3521 | 2818686.3483 |
| lstm_sequence | 1000 | 38.3391 | 33.8608 | 0.1570 | 1312642.9799 | 33.7258 | 30.8659 | -0.1842 | 897082.6281 |
| transformer_sequence | 1000 | 40.4629 | 35.4689 | 0.0611 | 2110421.1249 | 34.9107 | 31.8934 | -0.2688 | 1151347.1595 |
| gru_sequence | 1000 | 40.6465 | 35.7293 | 0.0525 | 2101379.1018 | 34.8661 | 31.9884 | -0.2656 | 1096462.5597 |

## Explanation quality

| Metric | Value |
| --- | ---: |
| faithfulness_deletion_delta | 116.8465 |
| stability_neighbor_cosine | 0.2108 |
| consistency_occlusion_lime_spearman | 1.0000 |
| robustness_noise_cosine | 0.9982 |
| sparsity_top3_share | 0.3893 |
| localization_last20pct_share | 0.3041 |

## Trustworthy behavior checks

| Metric | Value |
| --- | ---: |
| subgroup_fairness_mae_gap | 4.3732 |
| subgroup_fairness_max_group_mae | 21.2310 |
| confidence_uncertainty_error_spearman | -0.3058 |
| model_drift_wasserstein_mean | 0.0090 |
| adversarial_top_sensor_prediction_shift | 35.4321 |
| spurious_low_variance_attribution_share | 0.0000 |
| explanation_disagreement_one_minus_top5_jaccard | 0.0000 |

## XAI evaluation framework

| dimension | question | project_impact |
| --- | --- | --- |
| Predictive performance | Is the AI accurate? | Best model ridge test RMSE 21.868. |
| Interpretability | Can we understand why it predicted? | Global sensor, local sensor, temporal, and counterfactual explanations generated. |
| Faithfulness | Does explanation represent model behavior? | Deletion delta 116.847. |
| Stability | Do similar inputs produce similar explanations? | Neighbor cosine 0.211. |
| Robustness | Does explanation survive perturbations? | Noise cosine 0.998. |
| Fairness | Are decisions consistent across groups? | RUL subgroup MAE gap 4.373. |
| Confidence | Does reliability match confidence? | Uncertainty-error Spearman -0.306. |
| Actionability | Can explanation suggest what to do? | 8 maintenance-advisor rows generated. |
| Human usefulness | Does explanation improve decisions? | Engineer can inspect top sensors and high-risk engines first. |
| Domain impact | Does XAI produce practical benefit? | Maintenance prioritization combines RUL, risk, attribution, and counterfactual evidence. |

## Run scope

Deep sequence models enabled: True. Package-backed SHAP/LIME/DiCE enabled: True.
