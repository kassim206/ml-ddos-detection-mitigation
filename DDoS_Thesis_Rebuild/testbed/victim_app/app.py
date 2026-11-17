from datetime import datetime, timezone
from pathlib import Path
import csv

from flask import Flask, request

app = Flask(__name__)

LOG_FILE = Path("/app/logs/access_log.csv")

FIELDNAMES = [
    "timestamp_utc",
    "source_ip",
    "method",
    "path"
]


def log_request():
    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    write_header = (
        not LOG_FILE.exists()
        or LOG_FILE.stat().st_size == 0
    )

    with LOG_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "source_ip": request.remote_addr or "",
                "method": request.method,
                "path": request.path
            }
        )


@app.route("/", methods=["GET"])
def index():
    log_request()

    return (
        "Web Application Running - DDoS Testbed Target",
        200
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=80,
        debug=False
    )