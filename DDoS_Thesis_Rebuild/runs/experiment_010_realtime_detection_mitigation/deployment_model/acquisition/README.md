# Experiment 010 — Dataset B Acquisition

This directory contains the acquisition protocol and genuine controlled-laboratory
traffic runs used to develop the deployment-compatible detector.

Dataset B is separate from the selected CICIoT2023 subset used by Experiments
001–009. No distributional or feature-semantic equivalence between the two
datasets is assumed.

Each executed workload receives a unique immutable run directory.

A completed valid run is expected to retain:

- run_id.txt
- client_ground_truth.csv
- traffic.pcap
- victim_access_log.csv
- environment.txt
- docker_images.txt
- infrastructure_sha256.txt
- primary_evidence_sha256.txt
- reconciliation.json

Primary evidence must originate from actual execution. Empty, synthetic,
mock, reconstructed, or fabricated evidence must not be substituted.

Ground-truth class is determined from the workload that was deliberately
executed, not from packet features or model predictions.

Failed or incomplete executions are retained and explicitly classified as
invalid rather than silently converted into successful observations.

Feature extraction occurs only after provenance reconciliation passes.
Model training occurs only after the resulting Dataset B has been frozen and
split without leakage.
