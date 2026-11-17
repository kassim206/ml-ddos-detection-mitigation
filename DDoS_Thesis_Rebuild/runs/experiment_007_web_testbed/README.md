# Experiment 007 — Controlled HTTP Testbed Measurement

## Purpose

Experiment 007 measured a controlled local HTTP workload against the Dockerized victim web application.

The experiment sent genuine HTTP GET requests and recorded genuine client-side response measurements.

It was not a DDoS attack experiment, machine-learning detection experiment, or mitigation experiment.

## Configuration

- Target: `http://127.0.0.1:8080/`
- Requests attempted: 1000
- Concurrency: 10
- Request timeout: 5 seconds
- Method: HTTP GET
- Successful HTTP response definition: status 200–399

## Scope

Experiment 007 is separate from the offline CICIoT2023 classification experiments.

It did not:

- use CICIoT2023 observations;
- extract the 39 CICIoT2023 features from live traffic;
- perform model inference;
- integrate an intrusion detector;
- generate detector decisions;
- perform network mitigation; or
- execute iptables blocking.

The experiment therefore provides evidence only for the controlled HTTP workload and its observed response measurements.

## Retained Results

All 1000 attempted requests received HTTP 200 responses.

The retained measurements are:

- Responses received: 1000
- Transport failures: 0
- HTTP successful responses: 1000
- Total elapsed time: 3.461609135 seconds
- Observed wall-clock throughput: 288.882991 requests/second
- Mean response time: 34.315785 ms
- Median response time: 32.375097 ms
- Minimum response time: 7.251633 ms
- Maximum response time: 92.594998 ms
- p95 response time: 57.913035 ms
- p99 response time: 73.870532 ms
- Total response bytes: 45000

Percentiles use the nearest-rank method.

## Measurement Interpretation

Per-request response time was measured client-side using `time.perf_counter`, beginning immediately before `requests.get` and ending after response completion or a transport exception.

Observed throughput is the number of attempted requests divided by total client-side experiment wall-clock elapsed time.

These values characterize this controlled local HTTP workload only. They must not be interpreted as DDoS resistance, ML inference latency, attack-detection performance, or mitigation performance.

## Integrity

`traffic_log.csv` contains one retained record for each of the 1000 requests.

Its SHA-256 digest is:

`fbfa77799d36c036ed9c61b4dd653743d631a052555acd8c9f3ed68eddbefba8`

The same digest is recorded in `summary.json`.

No synthetic responses, synthetic latency values, synthetic status codes, or hardcoded experimental measurements are used.

## Artifacts

- `traffic_log.csv` — individual HTTP request measurements.
- `summary.json` — aggregate observed results.
- `config.json` — experiment methodology and scope.

The HTTP measurement client is:

`scripts/07_web_testbed_client.py`

The Docker victim infrastructure is located under:

`testbed/`
