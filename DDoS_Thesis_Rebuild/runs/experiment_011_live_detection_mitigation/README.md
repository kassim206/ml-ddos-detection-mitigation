# Experiment 011 — Live Detection and Application-Level Mitigation

## Purpose

Experiment 011 evaluates the frozen Experiment 010 deployment system in a controlled end-to-end Docker laboratory.

It establishes live packet acquisition, Dataset B feature extraction, frozen-model inference, detector decision generation, and application-level mitigation.

The experiment does not retrain the deployment model.

## End-to-End Chain

The retained system follows:

    controlled HTTP traffic
        -> live packet capture
        -> 100-packet feature windows
        -> frozen 20-feature pipeline
        -> frozen scaler
        -> frozen Logistic Regression model
        -> attack probability and detector decision
        -> shared mitigation state
        -> victim application enforcement
        -> retained client, detector, and victim evidence

## Frozen Deployment System

Experiment 011 uses the selected Experiment 010 system:

    Logistic Regression
    100-packet windows
    threshold = 0.5

The model and scaler are frozen inputs to the final live experiment.

## Protocol

The final protocol is retained under:

    protocol_v1/

It includes the experiment protocol, integrity rules, metric definitions, run plan, and SHA-256 integrity metadata.

The protocol defines the final experiment before interpretation of the result runs.

## Live Feature Pipeline

The live feature implementation is represented by:

    ../../scripts/11_live_feature_pipeline.py

Its frozen scientific copy is retained under:

    live_feature_pipeline_freeze_v1/

Feature-equivalence evidence is retained under:

    feature_pipeline_validation/

The validation establishes compatibility between the live extraction implementation and the Dataset B feature contract used by the frozen detector.

## Frozen Inference

Frozen inference is represented by:

    ../../scripts/11_frozen_model_inference.py

Its frozen scientific copy and integrity metadata are retained under:

    frozen_model_inference_freeze_v1/

## Detector Controller

The live detector controller is:

    ../../scripts/11_live_detector_controller.py

The detector captures relevant traffic, forms feature windows, invokes frozen inference, records detector events, and updates mitigation state according to the frozen experiment configuration.

## Mitigation Implementation

Mitigation state initialization is implemented by:

    ../../scripts/11_initialize_mitigation_state.py

The mitigation implementation is frozen under:

    real_mitigation_freeze_v1/

When enforcement is active, the victim application returns:

    HTTP 429 Too Many Requests

This is genuine application-level enforcement.

It is not iptables, firewall, router, source-IP, or upstream DDoS filtering.

## Final Conditions

Nine final result runs are retained.

### Benign

Runs:

    B01
    B02
    B03

Detector:

    enabled

Mitigation enforcement:

    enabled

All 3,000 retained requests received HTTP 200.

All 360 retained windows were classified as benign.

No mitigation action was activated.

### Attack Without Enforcement

Runs:

    A01
    A02
    A03

Detector:

    enabled

Mitigation enforcement:

    disabled

All 3,000 retained requests received HTTP 200.

All 360 retained windows were classified as attack.

This condition evaluates live detection while intentionally preventing enforcement.

### Attack With Enforcement

Runs:

    M01
    M02
    M03

Detector:

    enabled

Mitigation enforcement:

    enabled

Retained responses are:

| Run | HTTP 200 | HTTP 429 |
|---|---:|---:|
| M01 | 108 | 892 |
| M02 | 135 | 865 |
| M03 | 138 | 862 |

Across all three runs:

    HTTP 200 = 381
    HTTP 429 = 2619

All 3,000 transport attempts completed.

All retained requests occurring after enforcement activation received HTTP 429.

## Detection Evaluation

Detection evaluation is implemented by:

    ../../scripts/11_evaluate_detection_runtime.py

Retained evaluation evidence is under:

    evaluation_v1/

The pooled final evaluation contains:

    360 benign windows
    720 attack windows
    1080 total windows

The retained pooled confusion matrix is:

    [[360, 0],
     [0, 720]]

The pooled retained detection metrics are 1.0.

Single-class run-level metrics must not be interpreted in the same way as the pooled two-class evaluation.

Balanced accuracy is undefined where only one true class is present.

Under the retained fixed-label binary convention, correctly classified single-class runs have macro F1 equal to 0.5.

## Detector Timing

Retained mean feature-to-inference timing is approximately:

    Benign                         0.002629 s
    Attack without enforcement    0.001955 s
    Attack with enforcement       0.001967 s

The retained mean decision-to-enforcement transition for the mitigation runs is approximately:

    0.002484 s

These are defined detector-pipeline timing intervals.

They are not HTTP request latency or complete network end-to-end latency.

## Service Evaluation

Service and mitigation evaluation is implemented by:

    ../../scripts/11_evaluate_mitigation_service.py

Retained evidence is under:

    service_mitigation_evaluation_v1/

Exact process wall-clock timing was not retained for the final runs.

The retained analysis therefore uses archived request spans.

Request-span values must not be relabelled as exact process wall-clock runtime.

## Forensic Closure

Final reconciliation is implemented by:

    ../../scripts/11_forensic_reconciliation.py

Retained closure evidence is under:

    forensic_closure_v1/

The closure distinguishes supported claims from unsupported interpretations and explicitly records the remaining limitations.

## Experimental Unit

The experimental unit is the controlled run.

Feature windows are nested observations within runs.

The 1,080 pooled windows therefore do not represent 1,080 independent experimental replications.

## Docker Environment

The live Docker environment is defined by:

    ../../testbed/docker-compose-exp011.yml

The principal services are:

    victim_web_exp011
    detector_exp011
    benign_client_exp011
    attack_client_exp011

Complete operational documentation is provided in:

    ../../testbed/README.md

## Result Preservation

The final result runs are frozen scientific evidence.

Do not rerun traffic into the retained result directories.

New live traffic is a new experimental acquisition and should be stored separately.

## Interpretation Boundary

Experiment 011 supports:

- controlled live packet acquisition;
- deployment-compatible feature extraction;
- frozen-model inference;
- live detector decisions;
- controlled repeated run evaluation; and
- real application-level HTTP 429 mitigation.

It does not establish:

- firewall or iptables blocking;
- source-IP blacklisting;
- Internet-scale volumetric DDoS resistance;
- upstream traffic scrubbing;
- production deployment performance; or
- generalization to arbitrary real-world traffic.

See the root `README.md` for the complete thesis context.
