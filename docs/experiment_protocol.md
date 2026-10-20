# Experiment Protocol

## Research Questions

RQ1: Which course-aligned supervised model provides the strongest fraud-detection performance under severe class imbalance?

RQ2: Does cost-sensitive learning improve fraud recall and expected operational cost compared with default classification?

RQ3: Can an unsupervised anomaly-detection model identify fraud without using labels during training, and how does it compare with supervised models?

RQ4: Can a Genetic Algorithm learn better `APPROVE`, `MANUAL REVIEW`, and `BLOCK/ESCALATE` thresholds than manual or default thresholds?

RQ5: How sensitive are decisions to assumed missed-fraud, false-alarm, and manual-review costs?

RQ6: How does a chronological holdout compare with a stratified random split?

## Primary Split

Use a stratified 60/20/20 train/validation/test split with fixed random seed `582`.

- Training set: fit preprocessing, models, and internal model calibration.
- Validation set: model selection, threshold selection, GA fitness, cost-scenario tuning.
- Test set: final locked evaluation only after model and thresholds are frozen.

## Sensitivity Split

Sort by `Time` and use earlier transactions for training, the next period for validation, and the latest period for testing.

## Metrics

Primary metrics:

- Average precision/PR-AUC
- Fraud recall
- Fraud precision
- F1-score
- ROC-AUC
- Balanced accuracy
- Matthews Correlation Coefficient
- False-positive rate
- False-negative rate
- Operational cost
- Review volume
- Training and inference time

Accuracy may be shown only as secondary context.

## Baselines

1. Majority-class baseline.
2. Default threshold `0.50`.
3. Manual/grid-selected threshold from validation data.
4. GA-optimized three-action policy from validation predictions.

## Leakage Controls

- No preprocessing fit before splitting.
- No oversampling before splitting.
- GA uses validation predictions only.
- Test set is evaluated once after model, thresholds, and cost assumptions are frozen.
- Dataset labels are not used to train Isolation Forest.

## Cost Scenarios

Default cost assumptions are configured in `configs/default.yaml`. They are experimental assumptions only and must not be described as real bank costs.

