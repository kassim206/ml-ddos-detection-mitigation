#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path("runs/experiment_011_live_detection_mitigation")
RUN_ROOT = ROOT / "result_runs"
OUT = ROOT / "service_mitigation_evaluation_v1"
INPUT_MANIFEST = ROOT / "evaluation_v1" / "input_manifest.json"

RUNS = {
    "B01": "benign_with_detection_and_mitigation",
    "B02": "benign_with_detection_and_mitigation",
    "B03": "benign_with_detection_and_mitigation",
    "A01": "attack_detection_without_mitigation",
    "A02": "attack_detection_without_mitigation",
    "A03": "attack_detection_without_mitigation",
    "M01": "attack_detection_with_mitigation",
    "M02": "attack_detection_with_mitigation",
    "M03": "attack_detection_with_mitigation",
}

CONDITIONS = [
    "benign_with_detection_and_mitigation",
    "attack_detection_without_mitigation",
    "attack_detection_with_mitigation",
]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def dt(x):
    return datetime.fromisoformat(x)

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac

def describe(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }

OUT.mkdir(parents=True, exist_ok=True)

binding = json.loads(INPUT_MANIFEST.read_text())

if set(binding["result_runs"]) != set(RUNS):
    raise RuntimeError("input manifest run set mismatch")

integrity = {}

for run in RUNS:
    p = RUN_ROOT / run
    r1 = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=p,
        text=True,
        capture_output=True,
    )
    r2 = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS.sha256"],
        cwd=p,
        text=True,
        capture_output=True,
    )
    current = sha256(p / "SHA256SUMS")
    expected = binding["result_runs"][run]["manifest_sha256"]

    integrity[run] = {
        "artifact_manifest_verified": r1.returncode == 0,
        "manifest_hash_verified": r2.returncode == 0,
        "input_manifest_binding_verified": current == expected,
        "manifest_sha256": current,
    }

    if not all([
        integrity[run]["artifact_manifest_verified"],
        integrity[run]["manifest_hash_verified"],
        integrity[run]["input_manifest_binding_verified"],
    ]):
        raise RuntimeError(f"{run}: integrity verification failed")

per_run = {}

for run, condition in RUNS.items():
    p = RUN_ROOT / run
    client = read_csv(p / "client_ground_truth.csv")
    victim = read_csv(p / "access_log.csv")
    events = read_csv(p / "detector_events.csv")

    if len(client) != 1000 or len(victim) != 1000 or len(events) != 120:
        raise RuntimeError(f"{run}: unexpected row count")

    client_ids = [r["request_id"] for r in client]
    victim_ids = [r["request_id"] for r in victim]

    if len(set(client_ids)) != len(client_ids):
        raise RuntimeError(f"{run}: duplicate client request IDs")

    if len(set(victim_ids)) != len(victim_ids):
        raise RuntimeError(f"{run}: duplicate victim request IDs")

    if set(client_ids) != set(victim_ids):
        raise RuntimeError(f"{run}: request ID sets differ")

    client_by_id = {r["request_id"]: r for r in client}
    victim_by_id = {r["request_id"]: r for r in victim}

    mismatches = 0
    for request_id in client_by_id:
        if client_by_id[request_id]["status_code"] != victim_by_id[request_id]["response_status"]:
            mismatches += 1

    transport_success = sum(
        r["transport_success"].lower() == "true"
        for r in client
    )

    attempts = len(client)
    failed = attempts - transport_success

    starts = [dt(r["timestamp_start_utc"]) for r in client]
    finishes = [dt(r["timestamp_finish_utc"]) for r in client]
    latency = [float(r["elapsed_ms"]) for r in client]

    if not all(math.isfinite(x) and x >= 0 for x in latency):
        raise RuntimeError(f"{run}: invalid request latency")

    request_span = (max(finishes) - min(starts)).total_seconds()

    if request_span <= 0:
        raise RuntimeError(f"{run}: invalid request span")

    request_span_throughput = attempts / request_span

    attack_decisions = sum(r["predicted_label"] == "1" for r in events)
    action_requests = sum(
        r["mitigation_action_requested"].lower() == "true"
        for r in events
    )

    outcomes = Counter(r["enforcement_outcome"] for r in events)
    activated = [r for r in events if r["enforcement_outcome"] == "activated"]

    mitigation = {
        "model_attack_decisions": attack_decisions,
        "mitigation_action_requests": action_requests,
        "enforcement_outcomes": dict(outcomes),
        "activation_events": len(activated),
        "required_evidence_satisfied": None,
        "activation_window": None,
        "activation_probability": None,
        "decision_timestamp": None,
        "enforcement_completion_timestamp": None,
        "decision_to_enforcement_seconds": None,
        "requests_before_enforcement_completion": None,
        "requests_at_or_after_enforcement_completion": None,
        "post_enforcement_http_429": None,
        "post_enforcement_mitigation_active": None,
        "post_enforcement_failures": None,
    }

    if run.startswith("M"):
        if len(activated) != 1:
            raise RuntimeError(f"{run}: expected one activation")

        a = activated[0]

        if a["predicted_label"] != "1":
            raise RuntimeError(f"{run}: activation without attack decision")

        if a["mitigation_action_requested"].lower() != "true":
            raise RuntimeError(f"{run}: activation without action request")

        if not a["enforcement_completion_timestamp"]:
            raise RuntimeError(f"{run}: missing enforcement timestamp")

        completion = dt(a["enforcement_completion_timestamp"])

        before = [
            r for r in victim
            if dt(r["timestamp_utc"]) < completion
        ]
        after = [
            r for r in victim
            if dt(r["timestamp_utc"]) >= completion
        ]

        post_429 = sum(r["response_status"] == "429" for r in after)
        post_active = sum(
            r["mitigation_active"].lower() == "true"
            for r in after
        )
        failures = sum(
            r["response_status"] != "429"
            or r["mitigation_active"].lower() != "true"
            for r in after
        )

        decision_to_enforcement = (
            completion - dt(a["attack_decision_timestamp"])
        ).total_seconds()

        mitigation.update({
            "required_evidence_satisfied": (
                a["predicted_label"] == "1"
                and a["mitigation_action_requested"].lower() == "true"
                and a["enforcement_outcome"] == "activated"
                and bool(a["enforcement_completion_timestamp"])
                and len(after) > 0
                and failures == 0
            ),
            "activation_window": int(a["window_index"]),
            "activation_probability": float(a["attack_probability"]),
            "decision_timestamp": a["attack_decision_timestamp"],
            "enforcement_completion_timestamp": a["enforcement_completion_timestamp"],
            "decision_to_enforcement_seconds": decision_to_enforcement,
            "requests_before_enforcement_completion": len(before),
            "requests_at_or_after_enforcement_completion": len(after),
            "post_enforcement_http_429": post_429,
            "post_enforcement_mitigation_active": post_active,
            "post_enforcement_failures": failures,
        })

    per_run[run] = {
        "condition": condition,
        "request_attempts": attempts,
        "successful_transport_requests": transport_success,
        "failed_or_timed_out_requests": failed,
        "client_http_status_counts": dict(Counter(r["status_code"] for r in client)),
        "victim_http_status_counts": dict(Counter(r["response_status"] for r in victim)),
        "victim_mitigation_state_counts": dict(Counter(r["mitigation_active"] for r in victim)),
        "client_victim_status_mismatches": mismatches,
        "wall_clock_seconds": {
            "status": "NOT_SUPPORTED_FROM_ARCHIVED_ARTIFACTS",
            "value": None,
        },
        "archived_request_span_seconds": request_span,
        "archived_request_span_throughput_requests_per_second": request_span_throughput,
        "request_latency_ms": {
            "count": len(latency),
            "min": min(latency),
            "max": max(latency),
            "mean": statistics.mean(latency),
            "median": statistics.median(latency),
            "p95": percentile(latency, 0.95),
        },
        "mitigation": mitigation,
    }

condition_results = {}

for condition in CONDITIONS:
    ids = [r for r, c in RUNS.items() if c == condition]
    values = [per_run[r] for r in ids]

    statuses = Counter()
    victim_statuses = Counter()

    for x in values:
        statuses.update(x["client_http_status_counts"])
        victim_statuses.update(x["victim_http_status_counts"])

    condition_results[condition] = {
        "runs": ids,
        "n_runs": len(ids),
        "request_attempts": sum(x["request_attempts"] for x in values),
        "successful_transport_requests": sum(x["successful_transport_requests"] for x in values),
        "failed_or_timed_out_requests": sum(x["failed_or_timed_out_requests"] for x in values),
        "client_http_status_counts": dict(statuses),
        "victim_http_status_counts": dict(victim_statuses),
        "request_span_seconds_per_run": describe([
            x["archived_request_span_seconds"] for x in values
        ]),
        "request_span_throughput_per_run": describe([
            x["archived_request_span_throughput_requests_per_second"]
            for x in values
        ]),
        "latency_mean_ms_per_run": describe([
            x["request_latency_ms"]["mean"] for x in values
        ]),
        "latency_median_ms_per_run": describe([
            x["request_latency_ms"]["median"] for x in values
        ]),
        "latency_p95_ms_per_run": describe([
            x["request_latency_ms"]["p95"] for x in values
        ]),
    }

m_runs = [per_run[r] for r in ["M01", "M02", "M03"]]

post_total = sum(
    x["mitigation"]["requests_at_or_after_enforcement_completion"]
    for x in m_runs
)

post_429 = sum(
    x["mitigation"]["post_enforcement_http_429"]
    for x in m_runs
)

post_failures = sum(
    x["mitigation"]["post_enforcement_failures"]
    for x in m_runs
)

mitigation_summary = {
    "runs": ["M01", "M02", "M03"],
    "all_runs_satisfy_frozen_required_evidence": all(
        x["mitigation"]["required_evidence_satisfied"]
        for x in m_runs
    ),
    "successful_activation_events": sum(
        x["mitigation"]["activation_events"] for x in m_runs
    ),
    "post_enforcement_targeted_requests": post_total,
    "post_enforcement_http_429": post_429,
    "post_enforcement_failures": post_failures,
    "post_enforcement_http_429_fraction": post_429 / post_total,
    "decision_to_enforcement_seconds": describe([
        x["mitigation"]["decision_to_enforcement_seconds"]
        for x in m_runs
    ]),
}

result = {
    "experiment": "Exp011",
    "evaluation_version": 1,
    "experimental_unit": "run",
    "input_manifest_sha256": sha256(INPUT_MANIFEST),
    "integrity": integrity,
    "per_run": per_run,
    "by_condition": condition_results,
    "mitigation_summary": mitigation_summary,
    "service_metric_qualification": {
        "wall_clock_seconds": "NOT_SUPPORTED_FROM_ARCHIVED_ARTIFACTS",
        "archived_request_span_seconds": "Derived as last archived client request finish timestamp minus first archived client request start timestamp.",
        "request_span_throughput": "Derived as request attempts divided by archived request span. It is not relabeled as process wall-clock throughput.",
        "request_latency": "Derived from archived client elapsed_ms values.",
    },
    "replication_warning": "Requests are observations nested within runs and are not independent experimental replicates.",
    "scope_warning": "Mitigation is application-level HTTP 429 enforcement in the controlled Exp011 laboratory environment.",
}

(OUT / "mitigation_service_results.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)

with (OUT / "service_per_run.csv").open("w", newline="") as f:
    fields = [
        "run_id",
        "condition",
        "request_attempts",
        "successful_transport_requests",
        "failed_or_timed_out_requests",
        "http_200",
        "http_429",
        "archived_request_span_seconds",
        "request_span_throughput_requests_per_second",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_p95_ms",
        "mitigation_required_evidence_satisfied",
        "post_enforcement_requests",
        "post_enforcement_http_429",
        "post_enforcement_failures",
    ]

    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    for run in RUNS:
        x = per_run[run]
        m = x["mitigation"]

        w.writerow({
            "run_id": run,
            "condition": x["condition"],
            "request_attempts": x["request_attempts"],
            "successful_transport_requests": x["successful_transport_requests"],
            "failed_or_timed_out_requests": x["failed_or_timed_out_requests"],
            "http_200": x["client_http_status_counts"].get("200", 0),
            "http_429": x["client_http_status_counts"].get("429", 0),
            "archived_request_span_seconds": x["archived_request_span_seconds"],
            "request_span_throughput_requests_per_second": x["archived_request_span_throughput_requests_per_second"],
            "latency_mean_ms": x["request_latency_ms"]["mean"],
            "latency_median_ms": x["request_latency_ms"]["median"],
            "latency_p95_ms": x["request_latency_ms"]["p95"],
            "mitigation_required_evidence_satisfied": m["required_evidence_satisfied"],
            "post_enforcement_requests": m["requests_at_or_after_enforcement_completion"],
            "post_enforcement_http_429": m["post_enforcement_http_429"],
            "post_enforcement_failures": m["post_enforcement_failures"],
        })

print("ALL_RUN_INTEGRITY_VERIFIED", all(
    all([
        x["artifact_manifest_verified"],
        x["manifest_hash_verified"],
        x["input_manifest_binding_verified"],
    ])
    for x in integrity.values()
))

print()
print("PER_RUN")

for run in RUNS:
    x = per_run[run]
    print(
        run,
        "ATTEMPTS", x["request_attempts"],
        "SUCCESS", x["successful_transport_requests"],
        "STATUS", x["client_http_status_counts"],
        "SPAN", x["archived_request_span_seconds"],
        "SPAN_RPS", x["archived_request_span_throughput_requests_per_second"],
        "LAT_MEAN", x["request_latency_ms"]["mean"],
        "LAT_MEDIAN", x["request_latency_ms"]["median"],
        "LAT_P95", x["request_latency_ms"]["p95"],
    )

print()
print("BY_CONDITION")

for condition in CONDITIONS:
    print(condition, condition_results[condition])

print()
print("MITIGATION_SUMMARY", mitigation_summary)
print()
print("RESULT_SHA256", sha256(OUT / "mitigation_service_results.json"))
print("PER_RUN_CSV_SHA256", sha256(OUT / "service_per_run.csv"))
