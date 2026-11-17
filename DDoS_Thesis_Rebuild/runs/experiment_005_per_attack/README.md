# Experiment 005 — Original Attack-Family Stratified Evaluation

## Purpose

Experiment 005 evaluates the Experiment 009 Hybrid CNN-LSTM predictions after stratifying the held-out test observations by their original CICIoT2023 attack family.

This is not an eight-class attack-family classifier.

The classifier itself predicts four broad categories:

- Benign
- DDoS
- Recon
- Web

The original attack-family labels retained by Experiment 001 are used only to stratify those four-category predictions for analysis.

## Provenance

Predictions are read from:

`runs/experiment_009_hybrid_cnn_lstm/test_predictions.parquet`

Original attack-family provenance is read from:

`runs/experiment_001_dataset_char/cleaned_data/test/test_combined.csv`

Both artifacts contain 1,773,902 test observations.

Experiment 005 verifies sequential prediction row indices and exact row-by-row equality between the Experiment 009 `true_label` and the Experiment 001 `Category` before computing attack-family results.

## Metric Interpretation

`category_recall` is the proportion of observations from an original attack family that were correctly predicted as that family's four-class target category.

For example, recall for `DDoS-SYN_Flood` measures how often held-out DDoS-SYN observations were classified as `DDoS`.

It does not mean that the model predicted the specific label `DDoS-SYN_Flood`.

## Held-Out Results

| Original family | Target category | Test support | Correct category predictions | Category recall |
|---|---|---:|---:|---:|
| Benign_Final | Benign | 181840 | 0 | 0.000000 |
| DDoS-HTTP_Flood | DDoS | 0 | 0 | Not evaluated |
| DDoS-SlowLoris | DDoS | 0 | 0 | Not evaluated |
| DDoS-SYN_Flood | DDoS | 987587 | 986749 | 0.999151 |
| DDoS-UDP_Flood | DDoS | 589345 | 588949 | 0.999328 |
| Recon-PortScan | Recon | 13625 | 11672 | 0.856661 |
| SqlInjection | Web | 0 | 0 | Not evaluated |
| XSS | Web | 1505 | 423 | 0.281063 |

## Zero-Support Policy

`DDoS-HTTP_Flood`, `DDoS-SlowLoris`, and `SqlInjection` have zero observations in the held-out test partition.

Their recall and mean confidence are therefore undefined and are stored as null/blank in the retained artifacts.

They must be reported as **not evaluated**, not as having 0% recall.

This is distinct from `Benign_Final`, which has 181,840 held-out observations and an observed category recall of 0.0.

## Artifacts

- `attack_family_metrics.csv` — tabular family-stratified results.
- `attack_family_metrics.json` — results plus evaluation provenance and interpretation metadata.
- `config.json` — input provenance, row-alignment verification, and zero-support policy.

All reported values are derived from retained Experiment 009 predictions and Experiment 001 test-set provenance. No synthetic or mock metrics are used.
