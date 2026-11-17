# Experiment 010 — Deployment-Compatible Dataset B Detection

## Purpose

Experiment 010 establishes a deployment-compatible detection pipeline using a separate controlled-laboratory dataset referred to as Dataset B.

The experiment addresses the deployment gap between the offline 39-feature Dataset A representation and features that can be extracted directly and reproducibly from controlled packet captures.

Dataset B is not represented as equivalent to CICIoT2023 Dataset A.

## Experimental Chain

The Experiment 010 chain is:

    controlled Docker traffic
        -> PCAP acquisition
        -> deterministic feature extraction
        -> Dataset B
        -> run-level train/validation/test separation
        -> candidate model development
        -> validation-based model selection
        -> selected-system freeze
        -> one final held-out evaluation

## Traffic Classes

Dataset B contains two classes:

    Benign
    Attack

The Attack class represents the controlled high-rate HTTP workload used by the project's isolated Docker laboratory.

It must not be interpreted as representing the complete diversity of real-world DDoS traffic.

## Feature Representation

Dataset B uses 20 deployment-compatible features:

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

Feature extraction is implemented by:

    ../../scripts/10_extract_dataset_b_features.py

## Windowing

Two candidate non-overlapping packet-window regimes were evaluated:

    10 packets
    100 packets

Windows do not cross run boundaries.

Incomplete tail packets that do not form a complete window are discarded.

The final selected deployment system uses 100-packet windows.

## Run-Level Partitioning

The retained experiment uses controlled run identities rather than randomly mixing windows from the same run across model-development partitions.

The retained partition structure uses:

Training:

    B01-B04
    A01-A04

Validation:

    B05-B06
    A05-A06

Held-out test:

    B07-B08
    A07-A08

Windows are nested within these runs.

The run is the experimental unit.

## Candidate Models

Candidate development includes:

- Logistic Regression;
- Random Forest; and
- the retained hybrid candidate.

Model and window selection are performed using the validation partition.

The held-out test runs are not used for candidate selection.

## Selected Deployment System

The retained selected system is:

    Logistic Regression
    window size = 100 packets
    threshold = 0.5

Feature scaling is fitted using training data only.

The selected model, scaler, protocol, and associated metadata are frozen before final held-out evaluation.

## Held-Out Evaluation

The final held-out evaluation uses:

    B07
    B08
    A07
    A08

The pooled test set contains:

    240 Benign windows
    240 Attack windows
    480 total windows

The retained pooled confusion matrix is:

    [[240, 0],
     [0, 240]]

The retained pooled held-out metrics are 1.0 where defined by the evaluation.

This result is specific to the controlled Dataset B test distribution.

It is not evidence of perfect detection on arbitrary production traffic or on Dataset A.

## Deployment Relationship

Experiment 010 does not itself constitute the final live mitigation experiment.

Its selected and frozen deployment system becomes the detector used by Experiment 011.

Experiment 011 therefore tests deployment behavior without retraining or selecting a new model from the final live result runs.

## Principal Scripts

The Experiment 010 scripts are:

    ../../scripts/10_init_acquisition_run.sh
    ../../scripts/10_load_acquisition_run.sh
    ../../scripts/10_extract_dataset_b_features.py
    ../../scripts/10_train_select_deployment_model.py
    ../../scripts/10_evaluate_final_test.py

## Docker Configuration

The controlled acquisition environment is defined by:

    ../../testbed/docker-compose-exp010.yml

Operational details are documented in:

    ../../testbed/README.md

## Integrity

This directory contains multiple freeze and checksum artifacts associated with model development and final evaluation.

Freeze directories should remain intact even when they contain files whose contents also appear elsewhere in the repository.

Those copies establish historical experimental state and are therefore provenance rather than accidental duplication.

## Reproduction Warning

The retained final held-out evaluation has already been performed.

Do not rerun the final test and overwrite its artifacts merely to demonstrate that the code executes.

For independent replication, use a separate repository copy or separate output hierarchy.

## Interpretation Boundary

Experiment 010 supports a controlled-laboratory deployment-compatible detector.

It does not establish:

- Dataset B equivalence to CICIoT2023;
- production Internet traffic generalization;
- production DDoS resilience;
- firewall enforcement;
- source-IP blocking; or
- Internet-scale mitigation.

See the root `README.md` for the complete project context.
