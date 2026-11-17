import csv
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests


TARGET = os.environ.get(
    "TARGET_URL",
    "http://victim_web_exp010/"
)

REQUESTS = int(os.environ.get("REQUEST_COUNT", "25"))
DELAY_SECONDS = float(os.environ.get("DELAY_SECONDS", "0.1"))

RUN_ID = os.environ.get("RUN_ID")
if not RUN_ID:
    raise RuntimeError("RUN_ID environment variable is required")

LOG_FILE = Path("/app/logs/client_ground_truth.csv")

FIELDS = [
    "run_id",
    "request_id",
    "workload_type",
    "timestamp_start_utc",
    "timestamp_finish_utc",
    "target_url",
    "method",
    "status_code",
    "response_bytes",
    "elapsed_ms",
    "transport_success"
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

if LOG_FILE.exists():
    raise RuntimeError(
        f"Refusing to overwrite existing evidence: {LOG_FILE}"
    )

with LOG_FILE.open(
    "x",
    newline="",
    encoding="utf-8"
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=FIELDS
    )
    writer.writeheader()

    for _ in range(REQUESTS):

        request_id = str(uuid.uuid4())
        start_utc = utc_now()
        t0 = time.perf_counter()

        status_code = ""
        response_bytes = 0
        success = False

        headers = {"X-Request-ID": request_id}

        try:
            response = requests.get(
                TARGET,
                headers=headers,
                timeout=5
            )

            status_code = response.status_code
            response_bytes = len(response.content)
            success = True

        except requests.RequestException:
            pass

        elapsed_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        finish_utc = utc_now()

        writer.writerow(
            {
                "run_id": RUN_ID,
                "request_id": request_id,
                "workload_type": "benign_controlled_http",
                "timestamp_start_utc": start_utc,
                "timestamp_finish_utc": finish_utc,
                "target_url": TARGET,
                "method": "GET",
                "status_code": status_code,
                "response_bytes": response_bytes,
                "elapsed_ms": elapsed_ms,
                "transport_success": success
            }
        )

        handle.flush()

        time.sleep(DELAY_SECONDS)

print(
    f"Completed {REQUESTS} controlled benign requests. "
    f"Ground truth: {LOG_FILE}"
)
