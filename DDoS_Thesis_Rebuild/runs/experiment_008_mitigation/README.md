# Experiment 008 — Offline Model-Driven Mitigation-Policy Evaluation

## Purpose

Experiment 008 evaluates an offline mitigation decision policy using genuine held-out predictions produced by Experiment 009.

It does not implement or demonstrate live network mitigation.

## Prediction Source

The input is the retained Experiment 009 prediction artifact:

`runs/experiment_009_hybrid_cnn_lstm/test_predictions.parquet`

The retained source contains 1,773,902 held-out test observations.

Experiment 008 uses the existing Experiment 009 predictions and confidence values. It does not retrain the model or regenerate predictions.

The expected SHA-256 digest of the prediction artifact is:

`d8422a95daceb3431f7442f7a9eeab267982e5a1bdc9f6c52105d88b369b8a24`

## Decision Policy

For a selected confidence threshold, the offline policy is:

`BLOCK if predicted_label != "Benign" and confidence >= threshold; otherwise ALLOW`

`BLOCK` and `ALLOW` are offline policy-decision labels.

They do not mean that a network packet, connection, host, or IP address was actually blocked or allowed by a live enforcement mechanism.

Ground-truth labels are not used to make the BLOCK/ALLOW decision.

Ground truth is used only after the decision to evaluate policy outcomes.

## Thresholds

The retained evaluation covers exactly these thresholds:

- 0.50
- 0.60
- 0.70
- 0.80
- 0.85
- 0.90
- 0.95
- 0.99

The retained operating threshold is 0.85.

No interpolated threshold measurements are reported.

## Operating-Threshold Results

At threshold 0.85:

- Total observations: 1,773,902
- BLOCK decisions: 1,601,556
- ALLOW decisions: 172,346

For observations with malicious ground-truth categories:

- Malicious observations: 1,592,062
- Assigned BLOCK: 1,575,563
- Assigned ALLOW: 16,499
- Malicious BLOCK rate: 0.989637

For benign observations:

- Benign observations: 181,840
- Assigned BLOCK: 25,993
- Assigned ALLOW: 155,847
- Benign BLOCK rate: 0.142944

Category-specific BLOCK rates at threshold 0.85 are:

- DDoS: 0.998824
- Recon: 0.031266
- Web: 0.039867

These results show the behavior of the offline threshold policy on retained held-out predictions. They do not establish equivalent performance in live network deployment.

## Scope Limitations

Experiment 008 does not provide evidence of:

- live CICIoT2023-compatible feature extraction;
- live model inference;
- source-IP attribution;
- packet or connection enforcement;
- firewall integration;
- iptables enforcement;
- integration with the Experiment 007 HTTP workload;
- real-time intrusion detection; or
- end-to-end live network mitigation.

No validated pipeline was implemented that converts live network traffic into the 39-feature model representation and then maps a model decision back to a source IP or enforceable network flow.

## Integrity

The complete held-out prediction artifact is processed without sampling.

No synthetic predictions, synthetic confidence values, synthetic data, or mock traffic are used.

The retained output SHA-256 digests are:

- `mitigation_decisions.csv`: `b100dcb2fb7c5003c596f296e0ac12161955f83e0184b346bbd932f335883832`
- `threshold_analysis.csv`: `ba5c6b7b9859ef31473b30f91b2df3cb3918c924f98d4bf6fd4b08da58ba4584`
- `mitigation_metrics.json`: `4d99e0564b7a7cd897e0ca77ab481506fee99412dbc64b2dd582d84e3d48cd91`

## Artifacts

- `mitigation_decisions.csv` — row-level offline BLOCK/ALLOW policy decisions at threshold 0.85.
- `threshold_analysis.csv` — exact results for the eight evaluated thresholds.
- `mitigation_metrics.json` — operating-threshold metrics.
- `experiment_summary.json` — experiment summary and artifact hashes.
- `config.json` — provenance, policy definition, methodology, and scope metadata.

Academic reporting must describe Experiment 008 as an **offline model-driven mitigation-policy evaluation**, not as live mitigation.
