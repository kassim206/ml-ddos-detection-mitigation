# Experiment 009 — Hybrid CNN-LSTM

## Purpose

Experiment 009 evaluates the retained Hybrid CNN-LSTM classifier on the processed selected CICIoT2023 subset.

This is the project's principal hybrid neural-network experiment.

## Input Semantics

The model receives 39 processed CICIoT2023 features with input shape `(39, 1)`.

The 39 positions form a feature axis. They are not treated as verified chronological packet timesteps.

The released CSV representation used by this project does not retain an explicit timestamp that establishes consecutive CSV rows as a chronological sequence. Therefore, this experiment does not claim that the LSTM learns temporal dependencies between consecutive packets or CSV rows.

The LSTM is interpreted as modeling dependencies across convolutionally transformed feature positions.

## Architecture

The retained model is:

Conv1D → BatchNormalization → MaxPooling1D → Conv1D → BatchNormalization → MaxPooling1D → LSTM → Dense → Dropout → Softmax

The saved model is named `Hybrid_CNN_LSTM_FeatureAxis`.

## Training Outcome

Training did not complete the configured maximum number of epochs.

Seven complete epochs are retained in `training_history_recovered.csv`. A partial eighth epoch is excluded from the recovered history.

The retained best checkpoint corresponds to epoch 4 according to validation loss.

The recovered training-history artifact was reconstructed from retained terminal output and is not represented as a pristine callback-generated training-history file.

## Evaluation

Evaluation was performed separately using the retained best checkpoint.

The validation and test metrics, confusion matrices, and test predictions in this directory are retained evaluation artifacts.

The test prediction artifact contains direct inference results from the retained best checkpoint.

## Interpretation

Overall accuracy must be interpreted together with macro-averaged and per-class metrics because the selected dataset is strongly class-imbalanced.

The model's poor or absent predictions for some classes are retained as observed and must not be hidden, corrected manually, or replaced with synthetic results.

## Reproducibility Constraints

No synthetic training epochs, interpolated training values, fabricated predictions, or manually constructed evaluation metrics should be generated when retained artifacts are unavailable.

Experiment 009 must be described as a non-temporal feature-axis Hybrid CNN-LSTM. It must not be described as operating on verified chronological packet sequences.
