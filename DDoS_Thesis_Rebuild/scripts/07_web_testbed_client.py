import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


EXPERIMENT_ID = "007"
OUTPUT_DIR = Path("runs/experiment_007_web_testbed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTPUT_DIR / "traffic_log.csv"
CONFIG_PATH = OUTPUT_DIR / "config.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

FIELDNAMES = [
    "request_id",
    "started_at_utc",
    "finished_at_utc",
    "url",
    "status_code",
    "response_time_ms",
    "response_bytes",
    "http_success",
    "transport_success",
    "error"
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def nearest_rank_percentile(values, percentile):
    if not values:
        raise ValueError("Cannot calculate percentile from empty values")

    if not 0.0 < percentile <= 1.0:
        raise ValueError("Percentile must be in the interval (0, 1]")

    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    index = max(0, min(len(ordered) - 1, rank - 1))

    return float(ordered[index])


parser = argparse.ArgumentParser()

parser.add_argument(
    "--url",
    default="http://127.0.0.1:5000/api/resource"
)

parser.add_argument(
    "--requests",
    type=int,
    default=1000
)

parser.add_argument(
    "--concurrency",
    type=int,
    default=10
)

parser.add_argument(
    "--timeout",
    type=float,
    default=5.0
)

args = parser.parse_args()

if args.requests <= 0:
    raise ValueError("--requests must be greater than zero")

if args.concurrency <= 0:
    raise ValueError("--concurrency must be greater than zero")

if args.concurrency > args.requests:
    raise ValueError("--concurrency cannot exceed --requests")

if args.timeout <= 0:
    raise ValueError("--timeout must be greater than zero")

parsed_url = urlparse(args.url)

if parsed_url.scheme not in {"http", "https"}:
    raise ValueError("--url must use http or https")

if not parsed_url.hostname:
    raise ValueError("--url must contain a valid hostname")


def send_request(request_id):
    started_at_utc = utc_now()
    started = time.perf_counter()

    try:
        response = requests.get(
            args.url,
            timeout=args.timeout
        )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        finished_at_utc = utc_now()

        content = response.content

        return {
            "request_id": request_id,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "url": args.url,
            "status_code": int(response.status_code),
            "response_time_ms": float(elapsed_ms),
            "response_bytes": int(len(content)),
            "http_success": bool(
                200 <= response.status_code < 400
            ),
            "transport_success": True,
            "error": ""
        }

    except requests.RequestException as exc:
        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        finished_at_utc = utc_now()

        return {
            "request_id": request_id,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "url": args.url,
            "status_code": None,
            "response_time_ms": float(elapsed_ms),
            "response_bytes": 0,
            "http_success": False,
            "transport_success": False,
            "error": f"{type(exc).__name__}: {exc}"
        }


experiment_started_at_utc = utc_now()
experiment_start = time.perf_counter()

results = []

with ThreadPoolExecutor(
    max_workers=args.concurrency
) as executor:
    futures = {
        executor.submit(
            send_request,
            request_id
        ): request_id
        for request_id in range(args.requests)
    }

    for future in as_completed(futures):
        request_id = futures[future]

        try:
            result = future.result()
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected worker failure for request "
                f"{request_id}: {type(exc).__name__}: {exc}"
            ) from exc

        results.append(result)

experiment_elapsed = (
    time.perf_counter() - experiment_start
)

experiment_finished_at_utc = utc_now()

if len(results) != args.requests:
    raise AssertionError(
        f"Expected {args.requests} request results, "
        f"received {len(results)}"
    )

results.sort(
    key=lambda row: row["request_id"]
)

request_ids = [
    int(row["request_id"])
    for row in results
]

expected_request_ids = list(
    range(args.requests)
)

if request_ids != expected_request_ids:
    raise AssertionError(
        "Request identifiers are incomplete, duplicated, "
        "or out of sequence"
    )

for row in results:
    latency = float(
        row["response_time_ms"]
    )

    if not math.isfinite(latency):
        raise AssertionError(
            f"Non-finite latency for request "
            f"{row['request_id']}"
        )

    if latency < 0:
        raise AssertionError(
            f"Negative latency for request "
            f"{row['request_id']}"
        )

    response_bytes = int(
        row["response_bytes"]
    )

    if response_bytes < 0:
        raise AssertionError(
            f"Negative response byte count for request "
            f"{row['request_id']}"
        )

with LOG_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=FIELDNAMES
    )

    writer.writeheader()
    writer.writerows(results)

latencies = [
    float(row["response_time_ms"])
    for row in results
]

transport_successful = sum(
    1
    for row in results
    if row["transport_success"]
)

transport_failed = (
    len(results) - transport_successful
)

http_successful = sum(
    1
    for row in results
    if row["http_success"]
)

http_unsuccessful = (
    len(results) - http_successful
)

responses_received = sum(
    1
    for row in results
    if row["status_code"] is not None
)

total_response_bytes = sum(
    int(row["response_bytes"])
    for row in results
)

status_counts = {}

for row in results:
    if row["status_code"] is None:
        key = "NO_RESPONSE"
    else:
        key = str(
            int(row["status_code"])
        )

    status_counts[key] = (
        status_counts.get(key, 0) + 1
    )

error_type_counts = {}

for row in results:
    if not row["error"]:
        continue

    error_type = (
        row["error"].split(":", 1)[0]
    )

    error_type_counts[error_type] = (
        error_type_counts.get(
            error_type,
            0
        ) + 1
    )

if experiment_elapsed <= 0:
    raise AssertionError(
        "Experiment elapsed time must be positive"
    )

mean_latency = float(
    statistics.mean(latencies)
)

median_latency = float(
    statistics.median(latencies)
)

minimum_latency = float(
    min(latencies)
)

maximum_latency = float(
    max(latencies)
)

p95_latency = nearest_rank_percentile(
    latencies,
    0.95
)

p99_latency = nearest_rank_percentile(
    latencies,
    0.99
)

throughput = float(
    len(results) / experiment_elapsed
)

traffic_log_sha256 = sha256_file(
    LOG_PATH
)

summary = {
    "experiment_id": EXPERIMENT_ID,
    "status": "completed",
    "experiment_started_at_utc": experiment_started_at_utc,
    "experiment_finished_at_utc": experiment_finished_at_utc,
    "target_url": args.url,
    "requests_attempted": int(len(results)),
    "responses_received": int(responses_received),
    "transport_successful_requests": int(
        transport_successful
    ),
    "transport_failed_requests": int(
        transport_failed
    ),
    "http_successful_requests": int(
        http_successful
    ),
    "http_unsuccessful_requests": int(
        http_unsuccessful
    ),
    "transport_success_rate": float(
        transport_successful / len(results)
    ),
    "http_success_rate": float(
        http_successful / len(results)
    ),
    "total_elapsed_seconds": float(
        experiment_elapsed
    ),
    "observed_wall_clock_throughput_requests_per_second": throughput,
    "mean_response_time_ms": mean_latency,
    "median_response_time_ms": median_latency,
    "minimum_response_time_ms": minimum_latency,
    "maximum_response_time_ms": maximum_latency,
    "p95_response_time_ms": p95_latency,
    "p99_response_time_ms": p99_latency,
    "percentile_method": "nearest_rank",
    "total_response_bytes": int(
        total_response_bytes
    ),
    "status_counts": status_counts,
    "error_type_counts": error_type_counts,
    "traffic_log_sha256": traffic_log_sha256,
    "synthetic_responses": False,
    "synthetic_latency": False,
    "synthetic_status_codes": False,
    "hardcoded_measurements": False,
    "sampling": False
}

config = {
    "experiment_id": EXPERIMENT_ID,
    "status": "completed",
    "experiment_type": "controlled_http_testbed_measurement",
    "target_url": args.url,
    "requested_requests": int(
        args.requests
    ),
    "concurrency": int(
        args.concurrency
    ),
    "timeout_seconds": float(
        args.timeout
    ),
    "traffic_type": "controlled_authorized_http_load",
    "request_method": "GET",
    "http_success_definition": "HTTP status code 200 through 399",
    "response_time_measurement": "client-side wall-clock duration measured with time.perf_counter from immediately before requests.get until response completion or transport exception",
    "throughput_measurement": "requests attempted divided by total client-side experiment wall-clock elapsed time",
    "percentile_method": "nearest_rank",
    "ciciot2023_used": False,
    "offline_dataset_used": False,
    "model_inference_performed": False,
    "detector_integrated": False,
    "detector_decisions_generated": False,
    "ground_truth_used": False,
    "mitigation_performed": False,
    "synthetic_responses": False,
    "synthetic_latency": False,
    "synthetic_status_codes": False,
    "hardcoded_measurements": False,
    "sampling": False,
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "requests_version": requests.__version__,
    "output_files": {
        "traffic_log": str(LOG_PATH),
        "summary": str(SUMMARY_PATH),
        "config": str(CONFIG_PATH)
    }
}

with SUMMARY_PATH.open(
    "w",
    encoding="utf-8"
) as handle:
    json.dump(
        summary,
        handle,
        indent=2
    )

with CONFIG_PATH.open(
    "w",
    encoding="utf-8"
) as handle:
    json.dump(
        config,
        handle,
        indent=2
    )

print(
    json.dumps(
        summary,
        indent=2
    )
)