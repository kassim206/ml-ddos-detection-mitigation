#!/usr/bin/env python3

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np


FEATURE_PIPELINE = Path("scripts/11_live_feature_pipeline.py")
MODEL_FREEZE = Path(
    "runs/experiment_010_realtime_detection_mitigation/"
    "deployment_model/selected_system_freeze"
)

EXPECTED_FEATURE_PIPELINE_SHA256 = (
    "8ee96bfcfc2fe19fd4ddf26e2acaf13cc9ebd3518227b36a72ad428eefc50ad4"
)

EXPECTED_SELECTED_SYSTEM_SHA256 = (
    "907f2288a579ed849b9186bb9258b692868621ff1bb62c17a3c45ab367020eee"
)

EXPECTED_MODEL_SHA256 = (
    "5d07101d649141869c6da038b5f5c6d0f64a9dcf16b19c0d85ddc32eecb56fc9"
)

EXPECTED_SCALER_SHA256 = (
    "3ae7e8c2e8c669bc5535eec7a9e902ff0ee821fb3debab6189a847e0e7da8fa0"
)

WINDOW_SIZE = 100
THRESHOLD = 0.5
DLT_EN10MB = 1


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def packet_timestamp_utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def load_feature_pipeline():
    if sha256_file(FEATURE_PIPELINE) != EXPECTED_FEATURE_PIPELINE_SHA256:
        raise RuntimeError("Frozen live feature pipeline hash mismatch")

    spec = importlib.util.spec_from_file_location(
        "exp011_live_feature_pipeline",
        FEATURE_PIPELINE,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_frozen_model():
    selected_path = MODEL_FREEZE / "selected_system.json"
    model_path = MODEL_FREEZE / "model.joblib"
    scaler_path = MODEL_FREEZE / "scaler.joblib"

    if sha256_file(selected_path) != EXPECTED_SELECTED_SYSTEM_SHA256:
        raise RuntimeError("Frozen selected-system hash mismatch")

    if sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Frozen model hash mismatch")

    if sha256_file(scaler_path) != EXPECTED_SCALER_SHA256:
        raise RuntimeError("Frozen scaler hash mismatch")

    selected = json.loads(selected_path.read_text())

    if selected["selected_model"] != "logistic_regression":
        raise RuntimeError("Unexpected selected model")

    if int(selected["selected_window_size_packets"]) != WINDOW_SIZE:
        raise RuntimeError("Unexpected window size")

    if float(selected["selected_threshold"]) != THRESHOLD:
        raise RuntimeError("Unexpected threshold")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if list(model.classes_) != [0, 1]:
        raise RuntimeError("Unexpected model class ordering")

    return model, scaler


def read_exact(stream, count):
    data = bytearray()

    while len(data) < count:
        chunk = stream.read(count - len(data))

        if not chunk:
            if len(data) == 0:
                return None

            raise RuntimeError("Truncated PCAP stream")

        data.extend(chunk)

    return bytes(data)


def read_pcap_header(stream):
    header = read_exact(stream, 24)

    if header is None:
        raise RuntimeError("Missing PCAP global header")

    magic = header[:4]

    formats = {
        b"\xd4\xc3\xb2\xa1": ("<", 1_000_000),
        b"\xa1\xb2\xc3\xd4": (">", 1_000_000),
        b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000),
        b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000),
    }

    if magic not in formats:
        raise RuntimeError("Unsupported PCAP stream magic")

    endian, timestamp_divisor = formats[magic]
    fields = struct.unpack(endian + "IHHIIII", header)
    network = fields[6]

    if network != DLT_EN10MB:
        raise RuntimeError(
            f"Unexpected live datalink {network}; expected EN10MB"
        )

    return header, endian, timestamp_divisor


def packet_records(stream, archive, endian, timestamp_divisor):
    while True:
        header = read_exact(stream, 16)

        if header is None:
            return

        ts_sec, ts_fraction, captured_length, original_length = struct.unpack(
            endian + "IIII",
            header,
        )

        if captured_length <= 0:
            raise RuntimeError("Invalid captured packet length")

        if original_length <= 0:
            raise RuntimeError("Invalid original packet length")

        frame = read_exact(stream, captured_length)

        if frame is None:
            raise RuntimeError("Missing captured packet")

        archive.write(header)
        archive.write(frame)

        timestamp_units = ts_sec * timestamp_divisor + ts_fraction

        yield timestamp_units, frame


def atomic_write_state(path, state):
    temporary = path.with_suffix(".tmp")

    with temporary.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, path)


def activate_mitigation(path, probability, window_index):
    if not path.exists():
        raise RuntimeError("Mitigation state does not exist")

    state = json.loads(path.read_text())

    if state.get("active") is True:
        return "already_active", None

    if state.get("active") is not False:
        raise RuntimeError("Invalid mitigation state")

    state["active"] = True
    state["activated_at_utc"] = utc_now()
    state["attack_probability"] = float(probability)
    state["window_index"] = int(window_index)

    atomic_write_state(path, state)

    return "activated", utc_now()


def write_event(writer, handle, row):
    writer.writerow(row)
    handle.flush()
    os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--capture-pcap", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--capture-stats", type=Path, required=True)
    parser.add_argument(
        "--enforcement",
        choices=["on", "off"],
        required=True,
    )
    args = parser.parse_args()

    if args.events.exists():
        raise RuntimeError("Refusing to overwrite detector evidence")

    if args.capture_pcap.exists():
        raise RuntimeError("Refusing to overwrite detector PCAP evidence")

    if args.summary.exists():
        raise RuntimeError("Refusing to overwrite detector summary evidence")

    if args.capture_stats.exists():
        raise RuntimeError("Refusing to overwrite capture statistics evidence")

    if not args.state.exists():
        raise RuntimeError("Mitigation state must be initialized first")

    pipeline_module = load_feature_pipeline()
    model, scaler = load_frozen_model()

    feature_names = list(pipeline_module.FEATURES)

    if len(feature_names) != 20:
        raise RuntimeError("Unexpected frozen feature count")

    pipeline = pipeline_module.LiveWindowPipeline(
        window_size=WINDOW_SIZE
    )

    command = [
        "tcpdump",
        "-i",
        "eth0",
        "-U",
        "-s",
        "0",
        "-w",
        "-",
        "tcp",
        "port",
        "80",
    ]

    args.capture_stats.parent.mkdir(parents=True, exist_ok=True)
    capture_stats_handle = args.capture_stats.open(
        "x",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=capture_stats_handle,
        bufsize=0,
    )

    if process.stdout is None:
        raise RuntimeError("Failed to open tcpdump stream")

    args.events.parent.mkdir(parents=True, exist_ok=True)
    args.capture_pcap.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "window_index",
        "window_start_timestamp",
        "window_end_timestamp",
        "feature_completion_timestamp",
        "inference_start_timestamp",
        "inference_completion_timestamp",
        "attack_decision_timestamp",
        "attack_probability",
        "predicted_label",
        "enforcement_enabled",
        "mitigation_action_requested",
        "enforcement_outcome",
        "enforcement_completion_timestamp",
        "mitigation_activated",
        "inference_seconds_monotonic",
        "feature_to_inference_seconds_monotonic",
        "decision_to_enforcement_seconds_monotonic",
    ]

    eligible_in_window = 0
    window_start_timestamp = None
    raw_pcap_records_consumed = 0
    eligible_packets = 0
    complete_windows = 0
    stop_requested = False
    previous_normalized_timestamp_units = None
    timestamp_normalization_count = 0
    maximum_timestamp_adjustment_units = 0
    completion_status = "running"
    error_type = None
    error_message = None

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        pcap_header, endian, timestamp_divisor = read_pcap_header(
            process.stdout
        )

        with args.capture_pcap.open("xb") as capture, args.events.open(
            "x",
            newline="",
            encoding="utf-8",
        ) as f:
            capture.write(pcap_header)
            capture.flush()
            os.fsync(capture.fileno())

            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

            for raw_timestamp_units, frame in packet_records(
                process.stdout,
                capture,
                endian,
                timestamp_divisor,
            ):
                raw_pcap_records_consumed += 1

                normalized_timestamp_units = raw_timestamp_units

                if (
                    previous_normalized_timestamp_units is not None
                    and normalized_timestamp_units
                    < previous_normalized_timestamp_units
                ):
                    adjustment_units = (
                        previous_normalized_timestamp_units
                        - normalized_timestamp_units
                    )
                    normalized_timestamp_units = (
                        previous_normalized_timestamp_units
                    )
                    timestamp_normalization_count += 1
                    maximum_timestamp_adjustment_units = max(
                        maximum_timestamp_adjustment_units,
                        adjustment_units,
                    )

                previous_normalized_timestamp_units = (
                    normalized_timestamp_units
                )

                timestamp = normalized_timestamp_units / timestamp_divisor

                packet = pipeline_module.decode_packet(
                    timestamp,
                    frame,
                    DLT_EN10MB,
                )

                if packet is None:
                    continue

                eligible_packets += 1

                if eligible_in_window == 0:
                    window_start_timestamp = timestamp

                eligible_in_window += 1

                row = pipeline.admit(packet)

                if row is None:
                    continue

                if eligible_in_window != WINDOW_SIZE:
                    raise RuntimeError("Eligible packet accounting mismatch")

                window_end_timestamp = timestamp
                feature_completion_timestamp = utc_now()
                feature_completion_monotonic = time.perf_counter_ns()

                values = []

                for feature in feature_names:
                    value = row[feature]

                    if value is None:
                        if (
                            feature == "packet_rate"
                            and row["duration_seconds"] == 0
                        ):
                            value = 0.0
                        else:
                            raise RuntimeError(
                                f"Unexpected null feature {feature}"
                            )

                    value = float(value)

                    if not math.isfinite(value):
                        raise RuntimeError(
                            f"Nonfinite feature {feature}"
                        )

                    values.append(value)

                inference_start_timestamp = utc_now()
                inference_start_monotonic = time.perf_counter_ns()

                x = np.asarray([values], dtype=np.float64)
                transformed = scaler.transform(x)

                if not np.isfinite(transformed).all():
                    raise RuntimeError("Nonfinite scaled features")

                probability = float(
                    model.predict_proba(transformed)[0, 1]
                )

                if not math.isfinite(probability):
                    raise RuntimeError("Nonfinite attack probability")

                inference_completion_monotonic = time.perf_counter_ns()
                inference_completion_timestamp = utc_now()

                predicted = int(probability >= THRESHOLD)

                attack_decision_monotonic = time.perf_counter_ns()
                attack_decision_timestamp = utc_now()

                mitigation_action_requested = (
                    predicted == 1
                    and args.enforcement == "on"
                )

                activated = False
                enforcement_outcome = "not_requested"
                enforcement_completion_timestamp = ""
                decision_to_enforcement_seconds_monotonic = ""

                if mitigation_action_requested:
                    enforcement_start_monotonic = (
                        attack_decision_monotonic
                    )

                    enforcement_outcome, enforcement_timestamp = (
                        activate_mitigation(
                            args.state,
                            probability,
                            row["window_index"],
                        )
                    )

                    activated = enforcement_outcome == "activated"

                    if activated:
                        enforcement_completion_monotonic = (
                            time.perf_counter_ns()
                        )
                        enforcement_completion_timestamp = (
                            enforcement_timestamp
                        )
                        decision_to_enforcement_seconds_monotonic = (
                            enforcement_completion_monotonic
                            - enforcement_start_monotonic
                        ) / 1_000_000_000.0

                inference_seconds_monotonic = (
                    inference_completion_monotonic
                    - inference_start_monotonic
                ) / 1_000_000_000.0

                feature_to_inference_seconds_monotonic = (
                    inference_completion_monotonic
                    - feature_completion_monotonic
                ) / 1_000_000_000.0

                write_event(
                    writer,
                    f,
                    {
                        "window_index": row["window_index"],
                        "window_start_timestamp": packet_timestamp_utc(
                            window_start_timestamp
                        ),
                        "window_end_timestamp": packet_timestamp_utc(
                            window_end_timestamp
                        ),
                        "feature_completion_timestamp": (
                            feature_completion_timestamp
                        ),
                        "inference_start_timestamp": (
                            inference_start_timestamp
                        ),
                        "inference_completion_timestamp": (
                            inference_completion_timestamp
                        ),
                        "attack_decision_timestamp": (
                            attack_decision_timestamp
                        ),
                        "attack_probability": format(
                            probability,
                            ".17g",
                        ),
                        "predicted_label": predicted,
                        "enforcement_enabled": args.enforcement,
                        "mitigation_action_requested": str(
                            mitigation_action_requested
                        ).lower(),
                        "enforcement_outcome": enforcement_outcome,
                        "enforcement_completion_timestamp": (
                            enforcement_completion_timestamp
                        ),
                        "mitigation_activated": str(
                            activated
                        ).lower(),
                        "inference_seconds_monotonic": format(
                            inference_seconds_monotonic,
                            ".17g",
                        ),
                        "feature_to_inference_seconds_monotonic": format(
                            feature_to_inference_seconds_monotonic,
                            ".17g",
                        ),
                        "decision_to_enforcement_seconds_monotonic": (
                            ""
                            if (
                                decision_to_enforcement_seconds_monotonic
                                == ""
                            )
                            else format(
                                decision_to_enforcement_seconds_monotonic,
                                ".17g",
                            )
                        ),
                    },
                )

                capture.flush()
                os.fsync(capture.fileno())

                complete_windows += 1
                eligible_in_window = 0
                window_start_timestamp = None

                if stop_requested:
                    break

        completion_status = (
            "stopped"
            if stop_requested
            else "completed"
        )

    except BaseException as exc:
        completion_status = "failed"
        error_type = type(exc).__name__
        error_message = str(exc)
        raise

    finally:
        if process.poll() is None:
            process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        capture_stats_handle.flush()
        os.fsync(capture_stats_handle.fileno())
        capture_stats_handle.close()

        args.summary.parent.mkdir(parents=True, exist_ok=True)

        accounting_identity_pass = (
            eligible_packets
            == complete_windows * WINDOW_SIZE + eligible_in_window
        )
        tail_range_pass = 0 <= eligible_in_window < WINDOW_SIZE
        processing_integrity_pass = (
            completion_status in {"completed", "stopped"}
            and accounting_identity_pass
            and tail_range_pass
        )

        summary = {
            "raw_pcap_records_consumed": raw_pcap_records_consumed,
            "eligible_packets": eligible_packets,
            "complete_windows": complete_windows,
            "eligible_tail_packets": eligible_in_window,
            "window_size_packets": WINDOW_SIZE,
            "accounting_identity_pass": accounting_identity_pass,
            "tail_range_pass": tail_range_pass,
            "processing_integrity_pass": processing_integrity_pass,
            "timestamp_normalization_count": (
                timestamp_normalization_count
            ),
            "maximum_timestamp_adjustment_units": (
                maximum_timestamp_adjustment_units
            ),
            "timestamp_units_per_second": timestamp_divisor,
            "maximum_timestamp_adjustment_seconds": (
                maximum_timestamp_adjustment_units
                / timestamp_divisor
            ),
            "completion_status": completion_status,
            "error_type": error_type,
            "error_message": error_message,
            "tcpdump_returncode": process.returncode,
            "stop_requested": stop_requested,
            "completed_at_utc": utc_now(),
        }

        temporary = args.summary.with_suffix(".tmp")

        with temporary.open("x", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary, args.summary)


if __name__ == "__main__":
    main()
