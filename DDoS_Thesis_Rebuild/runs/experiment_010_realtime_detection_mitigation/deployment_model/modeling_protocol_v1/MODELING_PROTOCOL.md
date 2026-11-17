# Exp010 Dataset B Modeling Protocol v1

## Status

Prospective protocol frozen before Dataset B validation acquisition and
before model selection.

The purpose of this document is to prevent validation or test results from
silently changing preprocessing, model candidates, window sizes,
thresholds, metrics, or selection rules.

## Scientific scope

Dataset B is a separate controlled-laboratory dataset. It is not claimed to
be distributionally equivalent to CICIoT2023 (Dataset A).

The binary task is:

- Benign: predefined controlled benign HTTP workload.
- Attack: predefined controlled high-rate HTTP flooding workload.

The experiment does not establish arbitrary DDoS detection, production
Internet generalization, or CICIoT2023/live semantic equivalence.

## Partition isolation

Training:
B01-B04 and A01-A04.

Validation:
B05-B06 and A05-A06.

Test:
B07-B08 and A07-A08.

The acquisition run is the partition unit. Windows from one acquisition run
must never cross partitions. Random window-level splitting is prohibited.

The test partition must not participate in preprocessing fitting, candidate
selection, hyperparameter selection, window-size selection, threshold
selection, or early stopping.

## Feature assembly

The frozen label-blind extractor produces the 20 scientific features.

Class, partition, and logical run ID are attached only after extraction from
the frozen run plan. They are provenance/label metadata and are not model
features.

`window_index`, source IP, request ID, and run ID are not model features.

Candidate packet windows are fixed to 10 and 100 packets.

## Null and non-finite handling

The extractor may represent `packet_rate` as null when duration is zero.
That specific case is prospectively mapped to 0.0 before model input.

Any other null, NaN, positive infinity, or negative infinity is a hard
failure and must not be silently imputed.

## Preprocessing

StandardScaler is fit on training data only for logistic regression and the
hybrid neural candidate.

Validation and test data use the already-fitted training scaler.

Random forest does not require scaling.

All fitted preprocessing state must be saved and hashed.

## Candidate models

Exactly three model families are eligible:

1. Logistic regression.
2. Random forest.
3. Hybrid CNN-LSTM.

Their parameters are fixed in `modeling_protocol.json`.

The CNN-LSTM operates over the ordered feature axis. It must not be
described as learning temporal packet sequences.

The hybrid model is not required to win. Validation evidence determines
which frozen candidate is selected.

## Validation and threshold selection

Probability thresholds are selected only from the frozen grid:

0.10 through 0.90 in increments of 0.05.

The primary validation criterion is unweighted mean per-run balanced
accuracy across B05, B06, A05, and A06.

Because each acquisition run contains only one ground-truth class, the
per-run balanced-accuracy contribution is the recall of that run's present
class. A library balanced-accuracy function must not be blindly applied to
a single-class run.

This gives every acquisition run equal selection weight and avoids treating
correlated windows as independent experimental replicates.

The first tie-break is mean per-run macro-F1.

If still exactly tied, deployment complexity is preferred in this order:

logistic regression -> random forest -> hybrid CNN-LSTM.

If the same model remains tied across packet-window candidates, the
100-packet window is preferred.

Pooled-window metrics are reported but are not the primary independent
replication evidence.

Validation results may not be used to invent additional model families,
features, window sizes, preprocessing rules, or selection metrics.

## Test

The selected model, packet window, preprocessing state, and probability
threshold must be frozen before final test evaluation.

B07-B08 and A07-A08 are then evaluated once under the frozen system.

No model, threshold, preprocessing, or window reselection is permitted in
response to test performance.

## Interpretation

Windows from a common acquisition run are correlated and must not be
presented as thousands of independent experimental replicates.

Both per-run and pooled-window results must be reported.

Poor results are scientific results and must be preserved.

The defensible scope is controlled-laboratory discrimination between the
predefined benign HTTP workload and controlled high-rate HTTP flooding.
