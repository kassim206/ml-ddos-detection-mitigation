import csv
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


TARGET = os.environ.get(
    "TARGET",
    "http://victim_web_exp010/"
)

RUN_ID = os.environ.get("RUN_ID")

REQUEST_COUNT = int(
    os.environ.get("REQUEST_COUNT", "100")
)

CONCURRENCY = int(
    os.environ.get("CONCURRENCY", "10")
)

TIMEOUT_SECONDS = float(
    os.environ.get("TIMEOUT_SECONDS", "5")
)

OUTPUT = Path(
    "/app/logs/client_ground_truth.csv"
)

ALLOWED_HOST = "victim_web_exp010"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_configuration():
    if not RUN_ID:
        raise RuntimeError("RUN_ID is required")

    parsed = urlparse(TARGET)

    if parsed.scheme != "http":
        raise RuntimeError(
            "Only HTTP is permitted in this controlled testbed"
        )

    if parsed.hostname != ALLOWED_HOST:
        raise RuntimeError(
            f"Refusing target {parsed.hostname!r}; "
            f"only {ALLOWED_HOST!r} is permitted"
        )

    if REQUEST_COUNT < 1 or REQUEST_COUNT > 1000:
        raise RuntimeError(
            "REQUEST_COUNT must be between 1 and 1000"
        )

    if CONCURRENCY < 1 or CONCURRENCY > 25:
        raise RuntimeError(
            "CONCURRENCY must be between 1 and 25"
        )

    if OUTPUT.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing evidence: {OUTPUT}"
        )


def perform_request(sequence_number):
    request_id = str(uuid.uuid4())

    started_utc = utc_now()
    started_perf = time.perf_counter()

    status_code = ""
    response_bytes = 0
    transport_success = False
    error_type = ""

    headers = {"X-Request-ID": request_id}

    try:
        response = requests.get(
            TARGET,
            headers=headers,
            timeout=TIMEOUT_SECONDS
        )

        status_code = response.status_code
        response_bytes = len(response.content)
        transport_success = True

    except requests.RequestException as exc:
        error_type = type(exc).__name__

    elapsed_ms = (
        time.perf_counter() - started_perf
    ) * 1000.0

    finished_utc = utc_now()

    return {
        "run_id": RUN_ID,
        "sequence_number": sequence_number,
        "request_id": request_id,
        "workload_type":
            "controlled_high_rate_http",
        "timestamp_start_utc": started_utc,
        "timestamp_finish_utc": finished_utc,
        "target_url": TARGET,
        "method": "GET",
        "status_code": status_code,
        "response_bytes": response_bytes,
        "elapsed_ms": elapsed_ms,
        "transport_success":
            str(transport_success).lower(),
        "error_type": error_type
    }


def main():
    validate_configuration()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "run_id",
        "sequence_number",
        "request_id",
        "workload_type",
        "timestamp_start_utc",
        "timestamp_finish_utc",
        "target_url",
        "method",
        "status_code",
        "response_bytes",
        "elapsed_ms",
        "transport_success",
        "error_type"
    ]

    rows = []

    experiment_start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=CONCURRENCY
    ) as executor:

        futures = [
            executor.submit(
                perform_request,
                sequence_number
            )
            for sequence_number in range(
                1,
                REQUEST_COUNT + 1
            )
        ]

        for future in as_completed(futures):
            rows.append(future.result())

    experiment_seconds = (
        time.perf_counter() - experiment_start
    )

    # Restore deterministic request-number order
    # in the retained ground-truth file.
    rows.sort(
        key=lambda row: row["sequence_number"]
    )

    with OUTPUT.open(
        "x",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)
            f.flush()

    successes = sum(
        row["transport_success"] == "true"
        for row in rows
    )

    print(
        f"Completed {REQUEST_COUNT} controlled "
        f"high-rate HTTP attempts."
    )

    print(
        f"Transport successes: "
        f"{successes}/{REQUEST_COUNT}"
    )

    print(
        f"Client wall-clock seconds: "
        f"{experiment_seconds:.6f}"
    )

    print(
        "Ground truth: "
        "/app/logs/client_ground_truth.csv"
    )


if __name__ == "__main__":
    main()
