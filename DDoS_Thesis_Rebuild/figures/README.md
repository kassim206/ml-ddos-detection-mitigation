# Final Thesis Figures

## Purpose

This directory contains the final thesis-facing figures generated from retained experiment evidence.

The figures summarize the principal results across Dataset A, the controlled HTTP baseline, offline mitigation analysis, Dataset B, and the final live detection and mitigation experiment.

## Builder

The authoritative final figure builder is:

    ../scripts/14_build_final_thesis_figures.py

The builder derives plotted values from retained scientific evidence and final result sources.

The final figures must not be replaced with manually fabricated or synthetic result plots.

## Final Figure Set

### Figure 01 — Dataset A Partition

    figure_01_dataset_a_partition.png

Visualizes the retained Dataset A partition structure.

### Figure 02 — Dataset A Model Performance

    figure_02_dataset_a_model_performance.png

Compares retained Dataset A model performance.

Overall accuracy should be interpreted together with macro/class-level performance because Dataset A is imbalanced.

### Figure 03 — Attack-Family Performance

    figure_03_attack_family_performance.png

Visualizes retained attack-family-level evaluation.

Zero-support attack families are not represented as measured zero performance.

### Figure 04 — CPU Inference Runtime

    figure_04_cpu_inference_runtime.png

Visualizes retained CPU complete-test-set batch/chunk processing measurements.

The plotted values are not independent HTTP request latency or GPU latency.

### Figure 05 — HTTP Service Baseline

    figure_05_http_service_baseline.png

Visualizes the Experiment 007 controlled HTTP service baseline.

This figure does not represent ML detection or mitigation.

### Figure 06 — Offline Mitigation Policy

    figure_06_offline_mitigation_policy.png

Visualizes Experiment 008 offline mitigation-policy behavior.

The policy output is not evidence of live firewall enforcement.

### Figure 07 — Dataset B Held-Out Detection

    figure_07_dataset_b_heldout_detection.png

Visualizes the Experiment 010 held-out Dataset B detection result for the frozen selected deployment system.

The result is specific to the controlled Dataset B distribution.

### Figure 08 — Live Detection Runtime

    figure_08_live_detection_runtime.png

Visualizes Experiment 011 live detector evaluation and timing information.

Detector-pipeline timing must not be relabelled as HTTP request latency.

### Figure 09 — Live Mitigation Service

    figure_09_live_mitigation_service.png

Visualizes the Experiment 011 application-level mitigation and service results.

Mitigation is HTTP 429 application enforcement rather than firewall or iptables blocking.

## Interpretation Policy

Figures are reporting artifacts, not independent experimental evidence.

Their values must remain traceable to retained experiment artifacts.

The preferred chain is:

    experiment evidence
        -> figure builder
        -> final PNG
        -> thesis

Do not manually alter plotted values or redraw figures with values that cannot be traced to the retained experiments.

## Regeneration

The final figure builder is:

    python scripts/14_build_final_thesis_figures.py

Regeneration can change PNG bytes across different software, font, or rendering environments even when the underlying numerical values are unchanged.

For verification of the submitted frozen figures, use the retained final artifacts and associated scientific evidence rather than assuming byte-identical rendering on every platform.

See the root `README.md` for complete experimental context.
