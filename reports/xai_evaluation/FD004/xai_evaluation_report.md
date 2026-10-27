# XAI evaluation report

Subset: FD004
Window length: 30
Best model: ridge

## Predictive benchmark

| model | train_windows | validation_rmse | validation_mae | validation_r2 | validation_nasa_score | test_rmse | test_mae | test_r2 | test_nasa_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | 1000 | 23.2050 | 18.2920 | 0.6853 | 909885.1132 | 21.5676 | 17.0766 | 0.3765 | 1010601.9465 |
| hist_gradient_boosting | 1000 | 23.6386 | 17.7340 | 0.6734 | 366456.8417 | 20.5914 | 14.8606 | 0.4316 | 805596.2986 |
| random_forest | 1000 | 25.2440 | 19.4168 | 0.6276 | 363673.7258 | 21.1954 | 15.6998 | 0.3978 | 766400.9871 |
| mlp_neural_regressor | 1000 | 36.3266 | 30.6111 | 0.2288 | 2051656.3351 | 33.4471 | 28.2181 | -0.4996 | 2866649.2736 |
| lstm_sequence | 1000 | 37.1584 | 33.0360 | 0.1931 | 1508514.5204 | 32.4969 | 29.9804 | -0.4156 | 1106008.7038 |
| gru_sequence | 1000 | 39.7590 | 35.1582 | 0.0762 | 2791419.8787 | 34.1349 | 31.5585 | -0.5619 | 1467380.3591 |
| transformer_sequence | 1000 | 40.1158 | 35.7218 | 0.0595 | 2595374.7612 | 34.5682 | 32.3003 | -0.6018 | 1340240.4852 |

## Explanation quality

| Metric | Value |
| --- | ---: |
| faithfulness_deletion_delta | 97.0201 |
| stability_neighbor_cosine | 0.0572 |
| consistency_occlusion_lime_spearman | 1.0000 |
| robustness_noise_cosine | 0.9974 |
| sparsity_top3_share | 0.3520 |
| localization_last20pct_share | 0.2842 |

## Trustworthy behavior checks

| Metric | Value |
| --- | ---: |
| subgroup_fairness_mae_gap | 10.5111 |
| subgroup_fairness_max_group_mae | 26.1654 |
| confidence_uncertainty_error_spearman | -0.2592 |
| model_drift_wasserstein_mean | 0.0174 |
| adversarial_top_sensor_prediction_shift | 14.4723 |
| spurious_low_variance_attribution_share | 0.0000 |
| explanation_disagreement_one_minus_top5_jaccard | 0.0000 |

## XAI evaluation framework

| dimension | question | project_impact |
| --- | --- | --- |
| Predictive performance | Is the AI accurate? | Best model ridge test RMSE 21.568. |
| Interpretability | Can we understand why it predicted? | Global sensor, local sensor, temporal, and counterfactual explanations generated. |
| Faithfulness | Does explanation represent model behavior? | Deletion delta 97.020. |
| Stability | Do similar inputs produce similar explanations? | Neighbor cosine 0.057. |
| Robustness | Does explanation survive perturbations? | Noise cosine 0.997. |
| Fairness | Are decisions consistent across groups? | RUL subgroup MAE gap 10.511. |
| Confidence | Does reliability match confidence? | Uncertainty-error Spearman -0.259. |
| Actionability | Can explanation suggest what to do? | 8 maintenance-advisor rows generated. |
| Human usefulness | Does explanation improve decisions? | Engineer can inspect top sensors and high-risk engines first. |
| Domain impact | Does XAI produce practical benefit? | Maintenance prioritization combines RUL, risk, attribution, and counterfactual evidence. |

## Run scope

Deep sequence models enabled: True. Package-backed SHAP/LIME/DiCE enabled: True.
