# Experimental Evidence Directory

## Purpose

The `runs/` directory contains the retained scientific artifacts for the experimental programme supporting the thesis:

**Hybrid Machine Learning-Based Real-Time DDoS Detection and Mitigation for Web Applications**

These directories are not disposable application-output folders. They contain the evidence from which experimental results, final tables, final figures, and thesis claims are derived.

## Experiment Index

The retained experiment directories are:

- `experiment_001_dataset_char/` — Dataset A characterization, provenance, and deterministic partitioning.
- `experiment_002_preprocessing/` — Dataset A preprocessing and training-only feature scaling.
- `experiment_003_baselines/` — Random Forest and XGBoost baseline evaluation.
- `experiment_004_cnn_dense/` — retained CNN-Dense validation experiment.
- `experiment_005_per_attack/` — attack-family analysis using original attack provenance.
- `experiment_006_runtime/` — CPU batch inference benchmarking.
- `experiment_007_web_testbed/` — controlled HTTP service baseline.
- `experiment_008_mitigation/` — offline mitigation-policy evaluation.
- `experiment_009_hybrid_cnn_lstm/` — retained Hybrid CNN-LSTM evaluation.
- `experiment_010_realtime_detection_mitigation/` — Dataset B acquisition, feature extraction, model selection, freeze, and held-out evaluation.
- `experiment_011_live_detection_mitigation/` — controlled live detection and application-level mitigation.
- `thesis_evidence_reconciliation_v1/` — cross-experiment evidence reconciliation.

## Dataset Separation

Experiments 001–009 primarily operate on Dataset A, the selected CICIoT2023 CSV feature subset.

Experiment 010 introduces Dataset B, which is independently generated from controlled Docker packet captures and uses a separate 20-feature deployment-compatible representation.

Dataset B is not represented as statistically or semantically equivalent to Dataset A.

Experiment 011 uses the frozen deployment system selected in Experiment 010.

## Evidence Preservation

Files under `runs/` may include:

- manifests;
- configurations;
- partition metadata;
- preprocessing artifacts;
- fitted scalers;
- model files;
- checkpoints;
- predictions;
- confusion matrices;
- metrics;
- PCAP captures;
- extracted features;
- detector events;
- client ground truth;
- victim logs;
- mitigation state;
- runtime measurements;
- protocol definitions;
- frozen implementation copies;
- SHA-256 manifests;
- status files; and
- reconciliation evidence.

A duplicate file hash does not by itself mean that a file is redundant.

Identical files can occur intentionally where a particular implementation, model, protocol, or artifact was copied into a freeze directory to establish the exact state associated with an experiment.

## Frozen Evidence

Historical experiment evidence should not be modified merely to make the repository cleaner.

In particular, do not:

- replace weak results with newer results;
- rerun final tests and overwrite the retained outputs;
- alter frozen protocols after observing results;
- edit historical checksum manifests to conceal differences;
- delete failed validation evidence merely because a later validation passed; or
- deduplicate frozen provenance solely because two files have identical hashes.

## Re-execution

Scripts in `scripts/` can reproduce or inspect substantial parts of the experimental methodology.

However, commands that write experiment results should be executed in a separate repository copy when independent replication is intended.

A new training run, traffic acquisition, or live experiment is a new experiment and should not silently replace the retained historical evidence.

## Experimental Unit

For Dataset B and Experiment 011, packet windows are nested observations within controlled runs.

The experimental unit is the run.

Window-level observations must not be described as an equivalent number of independent experimental replications.

## Evidence Reconciliation

Cross-experiment reconciliation is retained in:

    thesis_evidence_reconciliation_v1/

Its purpose is to verify that the final scientific reporting remains consistent with the retained experiment artifacts.

Historical reconciliation and integrity artifacts are scientific provenance even when filenames contain terms such as `audit`, `status`, or `reconciliation`.

## Final Outputs

The thesis-facing tables and figures are stored outside `runs/`:

    ../tables/
    ../figures/

Their builders consume retained scientific evidence rather than manually entered result values.

See the root `README.md` for the complete experimental description.
