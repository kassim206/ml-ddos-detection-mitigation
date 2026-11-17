# Docker Experimental Testbed

## Overview

This directory contains the Dockerized controlled-laboratory infrastructure used by Experiments 007, 010, and 011 of the project:

**Hybrid Machine Learning-Based Real-Time DDoS Detection and Mitigation for Web Applications**

The three Docker workflows have different scientific purposes and must not be conflated.

Experiment 007 provides a controlled HTTP service baseline.

Experiment 010 provides controlled traffic acquisition, packet capture, Dataset B feature extraction, deployment-model development, and held-out evaluation.

Experiment 011 provides live packet capture, deployment-compatible feature extraction, frozen-model inference, detector decisions, and actual application-level HTTP mitigation.

All traffic-generation components are intended for the isolated authorized Docker laboratory only.

---

## Testbed Layout

The principal testbed structure is:

    testbed/
    ├── attack_client/
    │   ├── attack_client.py
    │   ├── Dockerfile
    │   └── requirements.txt
    ├── benign_client/
    │   ├── benign_client.py
    │   ├── Dockerfile
    │   └── requirements.txt
    ├── experiment_010/
    │   ├── capture/
    │   │   └── Dockerfile
    │   └── victim_app/
    │       ├── app.py
    │       ├── Dockerfile
    │       └── requirements.txt
    ├── experiment_011/
    │   ├── detector/
    │   │   ├── Dockerfile
    │   │   └── requirements.txt
    │   └── victim_app/
    │       ├── app.py
    │       ├── Dockerfile
    │       └── requirements.txt
    ├── logs/
    ├── victim_app/
    │   ├── app.py
    │   ├── Dockerfile
    │   └── requirements.txt
    ├── docker-compose.yml
    ├── docker-compose-exp010.yml
    ├── docker-compose-exp011.yml
    └── README.md

The generic `victim_app/` and `docker-compose.yml` belong to the Experiment 007 controlled HTTP baseline.

The `experiment_010/` components and `docker-compose-exp010.yml` belong to Dataset B acquisition.

The `experiment_011/` components and `docker-compose-exp011.yml` belong to the final live detection and mitigation experiment.

The benign and attack client implementations are shared by the controlled Exp010/Exp011 workflows.

---

## Requirements

The Docker workflows require Docker Engine and Docker Compose support.

Verify the installation with:

    docker --version
    docker compose version

Depending on the host Docker configuration, `sudo` may be required.

Where this README shows:

    docker compose

a system requiring elevated Docker permissions may use:

    sudo docker compose

instead.

The root machine-learning environment is separate from the individual Docker image environments.

---

# Experiment 007 — Controlled HTTP Baseline

## Purpose

Experiment 007 measures the behavior of a simple Dockerized Flask victim application under a controlled local HTTP workload.

It is a service baseline.

It does not perform:

- packet feature extraction;
- machine-learning inference;
- detector decision generation; or
- mitigation.

Stored experiment artifacts are located at:

    runs/experiment_007_web_testbed/

The client is:

    scripts/07_web_testbed_client.py

The Docker configuration is:

    testbed/docker-compose.yml

---

## Exp007 Architecture

The Exp007 Compose configuration contains a single Docker service:

    victim_web

The victim is built from:

    testbed/victim_app/

Container port:

    80

Host port:

    8080

Victim log mount:

    testbed/logs/victim/

The client runs from the host Python environment rather than as a service in the generic Compose configuration.

---

## Exp007 Retained Configuration

The retained Experiment 007 workload used:

- 1,000 requests;
- concurrency 10;
- timeout 5 seconds;
- HTTP GET; and
- target `http://127.0.0.1:8080/`.

The experiment records genuine client-side request measurements.

It does not generate synthetic latency or synthetic status-code measurements.

---

## Validate Exp007 Compose

From the repository root:

    docker compose -f testbed/docker-compose.yml config

Build the victim image:

    docker compose -f testbed/docker-compose.yml build

---

## Start Exp007 Victim

From the repository root:

    docker compose -f testbed/docker-compose.yml up --build -d

Check service status:

    docker compose -f testbed/docker-compose.yml ps

Check the victim manually:

    curl -i http://127.0.0.1:8080/

A successful request should return the victim application's HTTP response.

---

## Exp007 Client

The controlled client is:

    scripts/07_web_testbed_client.py

The retained historical Experiment 007 results already exist under:

    runs/experiment_007_web_testbed/

Do not overwrite those retained artifacts merely to verify the Docker image.

For independent reproduction, use a copy of the repository or otherwise preserve the historical result directory.

---

## Stop Exp007

    docker compose -f testbed/docker-compose.yml down

---

# Experiment 010 — Dataset B Acquisition and Deployment Model

## Purpose

Experiment 010 creates a deployment-compatible controlled-laboratory dataset from actual packet captures.

It was introduced because the 39-feature Dataset A representation cannot simply be assumed to be reconstructable from arbitrary live traffic with identical semantics.

Experiment 010 therefore establishes a separate Dataset B with an explicit live-extractable 20-feature contract.

Dataset B is not represented as equivalent to Dataset A.

---

## Exp010 Compose Configuration

Configuration:

    testbed/docker-compose-exp010.yml

The Compose project name is:

    exp010_testbed

The services are:

    victim_web_exp010
    benign_client_exp010
    capture_exp010
    attack_client_exp010

The isolated network is:

    exp010_net

---

## Exp010 Victim

Service:

    victim_web_exp010

Build context:

    testbed/experiment_010/victim_app/

The victim receives the controlled benign or high-rate HTTP workload.

Its live logs are mounted at:

    testbed/logs/experiment_010/victim/

---

## Exp010 Benign Client

Service:

    benign_client_exp010

Build context:

    testbed/benign_client/

Target:

    http://victim_web_exp010/

Retained workload parameters in the Compose configuration include:

    REQUEST_COUNT=1000
    DELAY_SECONDS=0.1

The logical run identifier is supplied through:

    EXP010_RUN_ID

Client output is mounted at:

    testbed/logs/experiment_010/client/

---

## Exp010 Attack Client

Service:

    attack_client_exp010

Build context:

    testbed/attack_client/

Target:

    http://victim_web_exp010/

Retained workload parameters include:

    REQUEST_COUNT=1000
    CONCURRENCY=10
    TIMEOUT_SECONDS=5

The logical run identifier is supplied through:

    EXP010_RUN_ID

The term `attack` in this controlled experiment refers to the project's high-rate HTTP workload class.

It must not be interpreted as an instruction to target external systems.

---

## Exp010 Capture Service

Service:

    capture_exp010

Build context:

    testbed/experiment_010/capture/

The service uses `tcpdump`.

It shares the victim service network namespace and captures:

    tcp port 80

The capture is written to:

    /capture/traffic.pcap

which is mounted to:

    testbed/logs/experiment_010/capture/

This PCAP is a genuine packet capture produced by the controlled testbed.

This is distinct from Dataset A files whose names contain `.pcap.csv`, which are feature CSVs rather than raw PCAP captures.

---

## Exp010 Logical Run Identifier

The Exp010 Compose configuration requires:

    EXP010_RUN_ID

A Compose command without that environment variable is intentionally rejected.

For non-result-bearing configuration validation, a placeholder can be used:

    EXP010_RUN_ID=VALIDATION docker compose -f testbed/docker-compose-exp010.yml config

For image construction:

    EXP010_RUN_ID=VALIDATION docker compose -f testbed/docker-compose-exp010.yml build

A placeholder validation ID must not be represented as an experimental result run.

---

## Exp010 Dataset B Features

Packet captures are converted into Dataset B features by:

    scripts/10_extract_dataset_b_features.py

The retained feature representation contains 20 features:

    packet_count
    duration_seconds
    header_length_mean
    ttl_mean
    fin_fraction
    syn_fraction
    rst_fraction
    psh_fraction
    ack_fraction
    packet_length_sum
    packet_length_min
    packet_length_max
    packet_length_mean
    packet_length_std
    packet_length_variance
    iat_mean
    iat_min
    iat_max
    iat_std
    packet_rate

Candidate packet-window sizes are:

    10
    100

Windows are non-overlapping.

Windows do not cross experimental-run boundaries.

Incomplete tail windows are discarded.

Population-statistic conventions used by the retained feature pipeline are part of the frozen Dataset B feature contract.

---

## Exp010 Acquisition Scripts

The principal acquisition-management scripts are:

    scripts/10_init_acquisition_run.sh
    scripts/10_load_acquisition_run.sh

Feature extraction is performed by:

    scripts/10_extract_dataset_b_features.py

Model selection is performed by:

    scripts/10_train_select_deployment_model.py

Final held-out evaluation is performed by:

    scripts/10_evaluate_final_test.py

These scripts are result-bearing scientific components.

Do not execute them against the frozen retained experiment simply to test Docker functionality.

---

## Exp010 Model Development

Candidate systems are compared using training and validation partitions.

The held-out test partition is not used for candidate selection.

The selected deployment system is:

    Logistic Regression
    100-packet windows
    threshold = 0.5

The feature scaler is fitted using training data only.

The selected system is frozen before final test evaluation.

---

## Exp010 Held-Out Test

The final retained held-out evaluation uses:

    B07
    B08
    A07
    A08

The pooled evaluation contains 480 windows:

    240 Benign
    240 Attack

The retained confusion matrix is:

    [[240, 0],
     [0, 240]]

The pooled retained metrics are perfect on this controlled held-out Dataset B evaluation.

This result applies to the retained controlled-laboratory distribution.

It must not be generalized automatically to arbitrary production or Internet traffic.

---

## Exp010 Reproduction Warning

The original Dataset B runs and the final test have already been completed and frozen.

Running new Docker traffic produces a new acquisition.

It does not recreate the historical artifact identity or historical SHA-256 hashes unless every relevant byte happens to be identical, which should not be assumed.

New acquisitions should therefore use a separate working copy or separate output location when the purpose is replication rather than modification of the retained scientific record.

---

# Experiment 011 — Live Detection and Mitigation

## Purpose

Experiment 011 deploys the frozen Experiment 010 selected detector into a controlled live Docker workflow.

It evaluates the chain:

    live controlled traffic
        -> packet capture
        -> Dataset B feature extraction
        -> frozen scaler
        -> frozen Logistic Regression inference
        -> detector decision
        -> mitigation state
        -> victim enforcement
        -> retained evidence

Experiment 011 therefore provides the project's end-to-end controlled-laboratory detection and mitigation evaluation.

---

## Exp011 Compose Configuration

Configuration:

    testbed/docker-compose-exp011.yml

Compose project name:

    exp011_testbed

Services:

    victim_web_exp011
    detector_exp011
    benign_client_exp011
    attack_client_exp011

Network:

    exp011_net

---

## Exp011 Required Environment Variables

The Compose configuration requires:

    EXP011_RUN_ID
    EXP011_ENFORCEMENT

`EXP011_RUN_ID` identifies the controlled logical run.

`EXP011_ENFORCEMENT` determines whether the detector is operating with mitigation enforcement enabled or disabled.

For configuration validation only:

    EXP011_RUN_ID=VALIDATION EXP011_ENFORCEMENT=off docker compose -f testbed/docker-compose-exp011.yml config

For image construction:

    EXP011_RUN_ID=VALIDATION EXP011_ENFORCEMENT=off docker compose -f testbed/docker-compose-exp011.yml build

These validation commands do not constitute an Experiment 011 result run.

---

## Exp011 Victim

Service:

    victim_web_exp011

Build context:

    testbed/experiment_011/victim_app/

Mounted live victim logs:

    testbed/logs/experiment_011/victim/

Mounted mitigation state:

    testbed/logs/experiment_011/state/

The victim is connected to `exp011_net`.

For compatibility with the shared client configuration it has the network alias:

    victim_web_exp010

The alias is intentional.

It allows the retained shared benign and attack clients to use their expected target hostname while communicating with the Experiment 011 victim.

---

## Exp011 Detector

Service:

    detector_exp011

Build context:

    repository root

Dockerfile:

    testbed/experiment_011/detector/Dockerfile

Working directory inside the detector:

    /repo

The detector launches:

    python scripts/11_live_detector_controller.py

The controller receives paths for:

    mitigation_state.json
    detector_events.csv
    detector_input.pcap
    detector_summary.json
    tcpdump_stats.txt

and receives the configured enforcement mode.

The detector shares the victim service network namespace using:

    network_mode: service:victim_web_exp011

This permits the detector capture process to observe the relevant victim-network traffic.

---

## Exp011 Read-Only Scientific Inputs

The detector mounts:

    ../scripts:/repo/scripts:ro
    ../runs:/repo/runs:ro

The `:ro` mount mode makes the repository scripts and retained experiment evidence read-only from inside the detector container.

Live detector output is written through:

    ./logs/experiment_011:/evidence

This separates live working evidence from the read-only retained scientific inputs used by the detector.

---

## Exp011 Benign Client

Service:

    benign_client_exp011

Target:

    http://victim_web_exp010/

Retained parameters include:

    REQUEST_COUNT=1000
    DELAY_SECONDS=0.1

Run identity is supplied through:

    EXP011_RUN_ID

Client evidence is written through:

    testbed/logs/experiment_011/client/

---

## Exp011 Attack Client

Service:

    attack_client_exp011

Target:

    http://victim_web_exp010/

Retained parameters include:

    REQUEST_COUNT=1000
    CONCURRENCY=10
    TIMEOUT_SECONDS=5

Run identity is supplied through:

    EXP011_RUN_ID

The attack client is a controlled laboratory workload generator.

It is intended only for the isolated authorized testbed.

---

## Frozen Live Feature Pipeline

The Experiment 011 live feature implementation is represented by:

    scripts/11_live_feature_pipeline.py

The frozen scientific copy is retained under the Experiment 011 freeze artifacts.

The live feature pipeline was validated against the Dataset B feature contract before final result collection.

The experiment retains replay/equivalence evidence under:

    runs/experiment_011_live_detection_mitigation/feature_pipeline_validation/

The purpose of this validation is to ensure that deployment-time feature extraction matches the feature semantics expected by the frozen Experiment 010 detector.

---

## Frozen Model Inference

Frozen inference is represented by:

    scripts/11_frozen_model_inference.py

The Experiment 011 detector uses the frozen Experiment 010 selected system rather than retraining a new model during live result collection.

The deployment model therefore remains fixed across the final Experiment 011 conditions.

---

## Mitigation State

Mitigation-state initialization is implemented by:

    scripts/11_initialize_mitigation_state.py

The detector controller updates the shared state when the configured detection condition is satisfied.

The Experiment 011 victim reads the shared mitigation state and applies the retained application-level enforcement behavior.

---

## Application-Level Enforcement

Experiment 011 mitigation is implemented at the HTTP application layer.

When enforcement is active, the victim responds with:

    HTTP 429 Too Many Requests

This is real application-level enforcement.

It is not equivalent to:

- iptables blocking;
- firewall filtering;
- source-IP blacklisting;
- router or switch ACL enforcement;
- upstream traffic scrubbing; or
- packet dropping before the application.

Documentation and thesis reporting must preserve this distinction.

---

## Final Experiment 011 Conditions

The retained final experiment contains nine result runs.

### Benign condition

    B01
    B02
    B03

Traffic class:

    Benign

Detector:

    enabled

Mitigation enforcement:

    enabled

All 3,000 retained benign requests received HTTP 200 responses.

All 360 retained benign feature windows were classified as benign.

No mitigation action was activated.

### Attack detection condition

    A01
    A02
    A03

Traffic class:

    Attack

Detector:

    enabled

Mitigation enforcement:

    disabled

All 3,000 requests received HTTP 200 responses.

All 360 retained attack feature windows were classified as attack.

This condition isolates live detection without application enforcement.

### Attack mitigation condition

    M01
    M02
    M03

Traffic class:

    Attack

Detector:

    enabled

Mitigation enforcement:

    enabled

Retained response counts are:

| Run | HTTP 200 | HTTP 429 |
|---|---:|---:|
| M01 | 108 | 892 |
| M02 | 135 | 865 |
| M03 | 138 | 862 |

Across these runs:

    HTTP 200 = 381
    HTTP 429 = 2619

All 3,000 transport attempts completed.

All retained post-enforcement requests received HTTP 429.

---

## Detection Evaluation

Final detection evaluation is produced by:

    scripts/11_evaluate_detection_runtime.py

Retained outputs are under:

    runs/experiment_011_live_detection_mitigation/evaluation_v1/

The pooled final evaluation contains:

    360 benign windows
    720 attack windows
    1080 total windows

Retained pooled confusion matrix:

    [[360, 0],
     [0, 720]]

The retained pooled overall detection metrics are 1.0.

Per-run and per-condition single-class metric interpretation differs from the pooled two-class evaluation.

Balanced accuracy is undefined for a run containing only one true class.

Under the retained fixed-label binary macro-F1 convention, such correctly classified single-class conditions produce macro F1 of 0.5.

This is a metric-definition consequence and not evidence that half of the windows were misclassified.

---

## Detector Runtime

Retained mean feature-to-inference timings are approximately:

    Benign:                         0.002629 s
    Attack, enforcement disabled: 0.001955 s
    Attack, enforcement enabled:  0.001967 s

Mean decision-to-enforcement timing for mitigation-enabled attack runs is approximately:

    0.002484 s

These measurements represent defined detector-pipeline intervals.

They are not:

- client HTTP response latency;
- total network round-trip latency;
- complete process wall-clock duration; or
- production-scale end-to-end mitigation latency.

---

## Service and Mitigation Evaluation

Final service evaluation is produced by:

    scripts/11_evaluate_mitigation_service.py

Retained outputs are under:

    runs/experiment_011_live_detection_mitigation/service_mitigation_evaluation_v1/

Exact process wall-clock timing was not archived for the final Experiment 011 result runs.

The retained service evaluation therefore uses request-span measurements derived from archived request timestamps.

These values must be reported as request spans.

They must not be relabelled as exact process wall-clock runtime.

The retained limitation is intentional and must not be repaired using invented or reconstructed timing.

---

## Forensic Reconciliation

Final Experiment 011 reconciliation is implemented by:

    scripts/11_forensic_reconciliation.py

Retained reconciliation evidence is under:

    runs/experiment_011_live_detection_mitigation/forensic_closure_v1/

The closure distinguishes supported claims, supported claims with documented limitations, and unsupported interpretations.

This protects the final reporting boundary between what the experiment actually demonstrates and broader claims that were not tested.

---

# Safe Docker Validation

The Docker configurations can be tested without generating a new scientific result.

From the repository root:

    docker compose -f testbed/docker-compose.yml config
    EXP010_RUN_ID=VALIDATION docker compose -f testbed/docker-compose-exp010.yml config
    EXP011_RUN_ID=VALIDATION EXP011_ENFORCEMENT=off docker compose -f testbed/docker-compose-exp011.yml config

Images can be built without running result traffic:

    docker compose -f testbed/docker-compose.yml build
    EXP010_RUN_ID=VALIDATION docker compose -f testbed/docker-compose-exp010.yml build
    EXP011_RUN_ID=VALIDATION EXP011_ENFORCEMENT=off docker compose -f testbed/docker-compose-exp011.yml build

These are the preferred Docker checks for repository validation because they do not replace retained experiment results.

---

# Why Full Historical Run Commands Are Not the Default Reproduction Procedure

The repository contains frozen scientific results.

Exp010 Dataset B acquisition and Exp011 final result collection are not ordinary application demonstrations that should be repeatedly executed against the same output directories.

Running the clients again can generate new logs, packet captures, timestamps, request ordering, and result artifacts.

New traffic is scientifically a new acquisition.

For that reason, the retained experiment should normally be verified from its archived evidence.

If independent replication is required, first make a separate copy of the repository or redirect the complete output hierarchy so the retained experiment remains unchanged.

---

# Experiment 011 Protocol Evidence

The retained protocol is located under:

    runs/experiment_011_live_detection_mitigation/protocol_v1/

It contains:

    experiment_protocol.json
    integrity_rules.json
    metric_definitions.json
    run_plan.json
    SHA256SUMS
    SHA256SUMS.sha256

These files define the final experiment before interpretation of the result runs.

---

# Experiment 011 Frozen Implementations

Frozen feature-pipeline evidence is retained under:

    runs/experiment_011_live_detection_mitigation/live_feature_pipeline_freeze_v1/

Frozen inference evidence is retained under:

    runs/experiment_011_live_detection_mitigation/frozen_model_inference_freeze_v1/

Frozen mitigation implementation evidence is retained under:

    runs/experiment_011_live_detection_mitigation/real_mitigation_freeze_v1/

These freeze directories are part of the scientific provenance and should not be removed merely because copies of active scripts also exist under `scripts/`.

The duplicate content has a provenance function: it records the implementation state associated with the frozen experiment.

---

# Logs Versus Frozen Results

Paths under:

    testbed/logs/

serve as Docker working/output locations.

Final scientific Experiment 010 and Experiment 011 evidence is retained under the corresponding directories in:

    runs/

Do not assume that a current working log is itself the authoritative frozen result simply because it has a similar filename.

Use the retained experiment manifests, run directories, freeze metadata, and checksum files when verifying historical results.

---

# Scope and Security Boundary

The Docker laboratory is a controlled local environment.

The attack workload is designed for the project's isolated victim application.

Do not change the target to a third-party system or any host for which you do not have explicit authorization.

The testbed does not demonstrate:

- Internet-scale volumetric attack generation;
- production firewall integration;
- upstream DDoS scrubbing;
- production reverse-proxy mitigation;
- distributed botnet behavior;
- source-IP reputation systems; or
- production availability guarantees.

Its scientific purpose is controlled reproducible evaluation of the implemented detection and application-level mitigation pipeline.

---

# Reporting Terminology

Use the following terminology when describing the Docker experiments.

Experiment 007:

    controlled HTTP service baseline

Experiment 010:

    controlled-laboratory Dataset B acquisition and deployment-model evaluation

Experiment 011:

    controlled-laboratory live detection and application-level HTTP mitigation

Experiment 011 enforcement:

    HTTP 429 application-level enforcement

Do not describe Experiment 011 enforcement as iptables or firewall blocking.

Do not describe the controlled high-rate HTTP workload as proof of Internet-scale volumetric DDoS.

Do not describe Dataset B as equivalent to CICIoT2023 Dataset A.

---

# Cleanup

After non-result-bearing Docker configuration or image-build validation, no scientific cleanup is required because no result traffic has been generated.

If containers were started only for infrastructure testing, inspect them before removal:

    docker compose -f testbed/docker-compose.yml ps -a

For the generic Exp007 stack:

    docker compose -f testbed/docker-compose.yml down

For an Exp010 working stack, the required run variable must remain defined when resolving the Compose file:

    EXP010_RUN_ID=VALIDATION docker compose -f testbed/docker-compose-exp010.yml down

For an Exp011 working stack:

    EXP011_RUN_ID=VALIDATION EXP011_ENFORCEMENT=off docker compose -f testbed/docker-compose-exp011.yml down

Do not delete retained `runs/` evidence as part of Docker cleanup.
