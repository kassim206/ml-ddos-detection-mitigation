# Experiment 002 — Preprocessing

## Status

Completed.

## Purpose

Experiment 002 applies the retained preprocessing pipeline to the deterministic Experiment 001 partitions.

The pipeline preserves the Experiment 001 train, validation, and test partition boundaries.

## Input and Output

Expected row counts are:

- Training: 7,702,608
- Validation: 1,236,682
- Test: 1,773,902
- Total: 10,713,192

Each processed observation contains 39 model input features plus the `Category` target column.

Processed outputs are retained as Parquet files.

## Preprocessing Procedure

Non-finite feature values are handled by replacing positive and negative infinity with missing values and then replacing missing feature values with zero.

A `StandardScaler` is fitted using the training partition only.

The same fitted scaler is then applied to the training, validation, and test partitions.

The scaler is retained as:

`scaler.pkl`

The preprocessing implementation uses chunked processing so that the complete retained partitions can be processed without sampling.

## Leakage Control

Validation and test observations are not used to fit the scaler.

The retained train/validation/test boundaries originate from Experiment 001 and are preserved through preprocessing.

Independent reconstruction verified the retained processed partitions against the documented preprocessing procedure.

## Provenance Limitation

`Original_Attack` is not retained in the processed Experiment 002 Parquet representation.

Original attack-family provenance remains available through the aligned Experiment 001 artifacts and is used where attack-family stratification is required.

## Integrity

No synthetic data or mock observations are generated.

No sampling is used.

The retained row order and category labels are preserved through preprocessing.

## Artifacts

- `train_processed.parquet` — processed training partition.
- `val_processed.parquet` — processed validation partition.
- `test_processed.parquet` — processed test partition.
- `scaler.pkl` — training-fitted StandardScaler.
- `preprocessing_log.json` — retained preprocessing metadata.
- `config.json` — preprocessing configuration and expected dimensions.
