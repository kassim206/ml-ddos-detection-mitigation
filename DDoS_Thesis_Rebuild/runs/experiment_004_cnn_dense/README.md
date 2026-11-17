# Experiment 004 — CNN-Dense Baseline

## Purpose

Experiment 004 evaluates a CNN-Dense neural-network baseline on the processed selected CICIoT2023 subset.

The saved Experiment 004 model does not contain an LSTM layer. Historical repository identifiers that described this experiment as `cnn_lstm` were therefore corrected to `cnn_dense`.

## Model Identity

The retained model is a CNN-Dense classifier operating on 39 processed input features.

The model uses a one-dimensional convolutional feature-processing stage followed by flattening and dense classification layers.

The 39 input positions represent features. They must not be interpreted as chronological packet timesteps or as a validated temporal sequence.

## Experimental Role

Experiment 004 is a neural-network baseline and is separate from Experiment 009.

Experiment 009 contains the actual Hybrid CNN-LSTM architecture.

Experiment 004 exhibited substantial class-level performance limitations. Those observed results are retained and must not be replaced by synthetic, manually adjusted, or inferred results.

## Evidence

The retained `best_model.keras` file is the saved model checkpoint.

The `training_config.json` file records the training configuration.

The `label_mapping.json` file records the class mapping.

The `diagnostics_best_checkpoint` directory contains read-only evaluation artifacts produced from the retained best checkpoint.

## Reporting Constraint

Experiment 004 must be described as CNN-Dense, not CNN-LSTM.

Numerical claims concerning Experiment 004 must be derived from retained experiment or diagnostic artifacts. Missing experimental results must not be replaced with fabricated fallback values.

## Retained Evaluation Scope

The retained best checkpoint has a reproducible validation-set diagnostic in `diagnostics_best_checkpoint`.

A historical baseline-comparison artifact containing CNN-Dense values that could not be traced to the retained checkpoint was excluded from the final evidence set. Those unsupported values are not used in the academic report.

The retained checkpoint's validation results are the supported Experiment 004 performance evidence. Validation results must not be presented as test-set results.
