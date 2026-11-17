# Experiment 006 — Runtime and Process-Memory Benchmark

## Purpose

Experiment 006 compares inference processing time, throughput, and process RSS for the retained classification models.

The benchmark covers:

- Random Forest
- XGBoost
- CNN-Dense
- Hybrid CNN-LSTM

## Benchmark Scope

The benchmark uses the complete processed Experiment 002 test partition containing 1,773,902 observations.

No test-set sampling is used.

Each model processes the complete test partition three times after a 512-observation warm-up.

The test data are read in chunks of 32,768 observations. Keras models use an internal prediction batch size of 512.

The benchmark was executed as a CPU inference benchmark.

## Timing Interpretation

`total_seconds` is the elapsed processing time for one complete pass over all 1,773,902 test observations.

`amortized_ms_per_sample` is calculated by dividing complete-pass elapsed time by the number of processed observations.

It is therefore an **amortized processing-time measure under chunked/batched inference**.

It must not be interpreted as:

- independent single-request latency;
- network request latency;
- end-to-end intrusion-detection latency;
- pure model forward-pass latency; or
- GPU inference latency.

`throughput_samples_per_second` similarly represents throughput achieved while processing the complete test partition using this benchmark methodology.

## Process-Memory Interpretation

Memory observations are process resident-set-size (RSS) measurements.

`peak_rss_mb` is the observed Python-process RSS during a measured repeat.

`peak_rss_increase_mb` is the difference between the repeat's starting process RSS and observed peak RSS.

These values are not model-only memory footprints and are not direct measurements of total system RAM consumption.

The benchmark does not report CPU-utilization percentages.

## Aggregate Results

| Model | Mean complete-pass time (s) | Mean amortized time (ms/observation) | Throughput (observations/s) |
|---|---:|---:|---:|
| Random Forest | 13.510782 | 0.007616 | 131295.29 |
| XGBoost | 14.191568 | 0.008000 | 124996.90 |
| CNN-Dense | 32.610156 | 0.018383 | 54397.23 |
| Hybrid CNN-LSTM | 103.747995 | 0.058486 | 17098.18 |

These values summarize the three retained measured repeats in `runtime_repeats.csv`.

`std_total_seconds` is the population standard deviation (`ddof=0`) of complete-pass elapsed time across the three retained measured repeats.

## Artifacts

- `runtime_repeats.csv` — individual measured benchmark repeats.
- `resource_usage.csv` — aggregate timing, throughput, and process-RSS statistics.
- `config.json` — benchmark methodology and interpretation metadata.

The retained runtime and resource values are measured outputs. Synthetic latency, synthetic resource values, and hardcoded benchmark results are not used.

## Reporting Constraint

Academic reporting must describe these results as complete-test-set, chunked/batched CPU inference measurements.

The amortized per-observation values must not be relabeled as independent per-request inference latency.
