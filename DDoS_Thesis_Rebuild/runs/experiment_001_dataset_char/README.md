# Experiment 001 — Dataset Characterization and Deterministic Partitioning

## Status

Completed.

## Purpose

Experiment 001 characterizes and deterministically partitions the selected CICIoT2023 CSV subset used in this project.

The selected subset contains 46 CSV files from eight source traffic/attack folders and 10,713,192 observations.

This experiment does not represent the complete set of CICIoT2023 attack types.

## Dataset Scope

The selected source folders are:

- Benign_Final
- DDoS-HTTP_Flood
- DDoS-SlowLoris
- DDoS-SYN_Flood
- DDoS-UDP_Flood
- Recon-PortScan
- SqlInjection
- XSS

The retained CSV representation contains 39 input features.

## Deterministic Partitioning

The source data were partitioned using a deterministic file-aware row-level procedure.

The retained split seed is:

`20261002`

The resulting partitions contain:

- Training: 7,702,608 observations
- Validation: 1,236,682 observations
- Test: 1,773,902 observations
- Total: 10,713,192 observations

The split manifest records source-file provenance, row ranges, partition assignments, and source SHA-256 digests.

Independent reconstruction verified that the retained cleaned train, validation, and test partitions match the documented source-row assignments.

## Integrity

All 46 selected source CSV files use the same 39-feature schema.

No synthetic or mock observations are introduced by this experiment.

Local SHA-256 digests establish the integrity of the retained local source files. They should not be described as official CICIoT2023 checksums.

## Artifacts

- `dataset_manifest.json` — selected-source provenance and integrity metadata.
- `split_manifest.json` — deterministic source-row partition assignments.
- `dataset_characterization.json` — dataset characterization results.
- `attack_distribution.json` — original attack-family distribution.
- `category_distribution.json` — mapped category distribution.
- `schema_discovery_report.json` — retained schema information.
- `cleaned_data/` — deterministic training, validation, and test CSV partitions.
- `config.json` — experiment metadata.

## Reporting Scope

Academic reporting must describe the input as a **selected CICIoT2023 subset**.

The retained `.pcap.csv` files are CSV feature representations derived from packet captures; this experiment does not claim to process raw PCAP files.
