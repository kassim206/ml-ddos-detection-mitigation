from flask import Flask, request
from datetime import datetime, timezone
import csv
import json
import os
import threading

app = Flask(__name__)

LOG_PATH = "/app/logs/access_log.csv"
STATE_PATH = "/app/state/mitigation_state.json"
LOG_LOCK = threading.Lock()

FIELDS = [
    "timestamp_utc",
    "source_ip",
    "method",
    "path",
    "request_id",
    "mitigation_active",
    "response_status",
]


def mitigation_active():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except FileNotFoundError:
        return False

    if not isinstance(state, dict):
        raise RuntimeError("Invalid mitigation state")

    value = state.get("active")

    if not isinstance(value, bool):
        raise RuntimeError("Invalid mitigation active value")

    return value


def write_log(row):
    with LOG_LOCK:
        write_header = (
            not os.path.exists(LOG_PATH)
            or os.path.getsize(LOG_PATH) == 0
        )

        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)

            if write_header:
                writer.writeheader()

            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())


@app.route("/", methods=["GET"])
def index():
    active = mitigation_active()
    status = 429 if active else 200

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_ip": request.remote_addr or "",
        "method": request.method,
        "path": request.path,
        "request_id": request.headers.get("X-Request-ID", ""),
        "mitigation_active": str(active).lower(),
        "response_status": status,
    }

    write_log(row)

    if active:
        return "MITIGATED\n", 429

    return "OK\n", 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=80,
        threaded=True,
    )
