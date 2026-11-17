# Experiment 003 — Random Forest and XGBoost Baselines

## Status

Completed.

## Purpose

Experiment 003 trains and evaluates Random Forest and XGBoost baseline classifiers using the processed selected CICIoT2023 subset.

The classifiers operate on the 39-feature representation produced by Experiment 002.

## Training Input

Training observations:

`7,702,608`

Test observations:

`1,773,902`

The retained training memmap contains the complete Experiment 002 training partition.

Independent reconstruction verified the retained training feature and label arrays against the Experiment 002 training data.

## Random Forest Results

Held-out test performance:

- Accuracy: 0.9950369299
- Macro precision: 0.8840855841
- Macro recall: 0.6371391253
- Macro F1: 0.6823717529

## XGBoost Results

Held-out test performance:

- Accuracy: 0.9950059248
- Macro precision: 0.9634883941
- Macro recall: 0.6248751779
- Macro F1: 0.6692133178

Because the selected dataset is strongly class-imbalanced, overall accuracy must not be interpreted independently of macro and class-level performance.

## Artifact Verification

The retained model artifacts reproduce their corresponding held-out predictions and metrics.

The retained confusion matrices reconcile mathematically with the reported metrics.

The training memmap provenance was independently checked against the complete Experiment 002 training partition.

The retained artifacts establish the training input and evaluation behavior of the saved models. They do not constitute cryptographic proof of every historical in-memory operation performed during the original model-fitting process.

## Integrity

No synthetic or mock data are used to generate the reported baseline results.

No synthetic or manually constructed evaluation metrics are reported.

## Artifacts

- `model_rf.pkl` — retained Random Forest model.
- `model_xgb.pkl` — retained XGBoost model.
- `metrics_rf.json` — Random Forest test metrics.
- `metrics_xgb.json` — XGBoost test metrics.
- `confusion_matrix_rf.csv` — Random Forest test confusion matrix.
- `confusion_matrix_xgb.csv` — XGBoost test confusion matrix.
- `experiment_003_summary.json` — retained experiment summary.
- `training_memmap/` — retained training feature and label arrays.
- `config.json` — experiment metadata.
