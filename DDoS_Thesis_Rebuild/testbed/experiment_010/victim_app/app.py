from flask import Flask, request
from datetime import datetime, timezone
import csv
import os
import threading

app = Flask(__name__)

LOG_PATH = "/app/logs/access_log.csv"
LOG_LOCK = threading.Lock()

FIELDS = [
    "timestamp_utc",
    "source_ip",
    "method",
    "path",
    "request_id",
]


@app.route("/", methods=["GET"])
def index():
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_ip": request.remote_addr or "",
        "method": request.method,
        "path": request.path,
        "request_id": request.headers.get("X-Request-ID", ""),
    }

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

    return "OK\n", 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=80,
        threaded=True,
    )
