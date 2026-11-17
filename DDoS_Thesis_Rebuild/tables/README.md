# Final Thesis Tables

## Purpose

This directory contains the final thesis-facing tables generated from retained experimental evidence.

The CSV files are reporting outputs.

They are not manually entered substitutes for the underlying experiment artifacts.

## Builder

The authoritative final table builder is:

    ../scripts/13_build_final_thesis_tables.py

The builder reads retained experiment evidence and produces the final table set.

## Final Table Set

### Table 01 — Dataset A Composition and Split

    table_01_dataset_a_composition_split.csv

Summarizes the selected CICIoT2023 Dataset A composition and retained train/validation/test partitioning.

### Table 02 — Dataset A Model Performance

    table_02_dataset_a_model_performance.csv

Summarizes retained Dataset A model performance, including classical baselines and the retained neural-model evaluations with their appropriate evaluation scope.

### Table 03 — Attack-Family Performance

    table_03_attack_family_performance.csv

Reports attack-family-level behavior derived from original attack provenance and retained model predictions.

Zero-support families are treated as not evaluated rather than assigned fabricated performance.

### Table 04 — CPU Batch Inference Runtime

    table_04_cpu_batch_inference_runtime.csv

Reports retained complete-test-set CPU batch/chunk inference measurements.

These values are not HTTP request latency or GPU latency.

### Table 05 — Controlled HTTP Service Baseline

    table_05_controlled_http_service_baseline.csv

Reports the Experiment 007 controlled HTTP service measurement.

Experiment 007 does not perform ML detection or mitigation.

### Table 06 — Offline Mitigation Policy

    table_06_offline_mitigation_policy.csv

Reports Experiment 008 offline mitigation-selection behavior.

The table does not represent actual firewall or application enforcement.

### Table 07 — Dataset B Held-Out Detection

    table_07_dataset_b_heldout_detection.csv

Reports the frozen Experiment 010 selected system on the held-out Dataset B test runs.

The result applies to the controlled Dataset B distribution.

### Table 08 — Live Detection Runtime

    table_08_live_detection_runtime.csv

Reports Experiment 011 pooled live detection results and condition-level detector timing.

Pooled overall metrics are reported on the pooled row.

Condition rows are used for their defined timing measurements and should not be interpreted as independent pooled two-class evaluations.

### Table 09 — Live Mitigation Service Impact

    table_09_live_mitigation_service_impact.csv

Reports Experiment 011 service and application-level mitigation results.

The retained timing quantity is request span.

It must not be relabelled as exact process wall-clock runtime.

## Integrity Metadata

This directory also retains:

    TABLES_CLOSURE_STATUS.json
    TABLES_SHA256SUMS
    TABLES_SHA256SUMS.sha256

These files document the frozen final table set and its integrity state.

## Editing Policy

Do not manually alter final CSV values to improve presentation or results.

Scientific corrections must originate from the underlying evidence and builder logic.

When the thesis cites a numerical result, the preferred traceability chain is:

    experiment evidence
        -> final table builder
        -> table CSV
        -> thesis

See the root `README.md` for the complete experiment descriptions.
