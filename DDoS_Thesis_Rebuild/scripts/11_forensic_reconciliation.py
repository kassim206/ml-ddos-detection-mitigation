#!/usr/bin/env python3

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path("runs/experiment_011_live_detection_mitigation")
RUN_ROOT = ROOT / "result_runs"
S6 = ROOT / "evaluation_v1" / "detection_runtime_results.json"
S7 = ROOT / "service_mitigation_evaluation_v1" / "mitigation_service_results.json"

RUNS = {
    "B01": {"condition": "benign_with_detection_and_mitigation", "true_label": 0, "enforcement": "on"},
    "B02": {"condition": "benign_with_detection_and_mitigation", "true_label": 0, "enforcement": "on"},
    "B03": {"condition": "benign_with_detection_and_mitigation", "true_label": 0, "enforcement": "on"},
    "A01": {"condition": "attack_detection_without_mitigation", "true_label": 1, "enforcement": "off"},
    "A02": {"condition": "attack_detection_without_mitigation", "true_label": 1, "enforcement": "off"},
    "A03": {"condition": "attack_detection_without_mitigation", "true_label": 1, "enforcement": "off"},
    "M01": {"condition": "attack_detection_with_mitigation", "true_label": 1, "enforcement": "on"},
    "M02": {"condition": "attack_detection_with_mitigation", "true_label": 1, "enforcement": "on"},
    "M03": {"condition": "attack_detection_with_mitigation", "true_label": 1, "enforcement": "on"},
}

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def dt(x):
    return datetime.fromisoformat(x)

def cm_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    tnr = tn / (tn + fp) if tn + fp else None
    tpr = tp / (tp + fn) if tp + fn else None
    balanced = (tnr + tpr) / 2 if tnr is not None and tpr is not None else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tpr
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    neg_precision = tn / (tn + fn) if tn + fn else None
    neg_recall = tnr
    neg_f1 = (
        2 * neg_precision * neg_recall / (neg_precision + neg_recall)
        if neg_precision is not None and neg_recall is not None and neg_precision + neg_recall
        else 0.0
    )
    pos_f1 = f1 if f1 is not None else 0.0
    macro_f1 = (neg_f1 + pos_f1) / 2
    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }

raw = {}
overall = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
condition_counts = {}

for run, spec in RUNS.items():
    p = RUN_ROOT / run
    client = read_csv(p / "client_ground_truth.csv")
    victim = read_csv(p / "access_log.csv")
    events = read_csv(p / "detector_events.csv")
    summary = json.loads((p / "detector_summary.json").read_text())
    state = json.loads((p / "mitigation_state.json").read_text())

    if len(client) != 1000:
        raise RuntimeError(f"{run}: client rows")
    if len(victim) != 1000:
        raise RuntimeError(f"{run}: victim rows")
    if len(events) != 120:
        raise RuntimeError(f"{run}: event rows")

    if {r["request_id"] for r in client} != {r["request_id"] for r in victim}:
        raise RuntimeError(f"{run}: request ID mismatch")

    if [int(r["window_index"]) for r in events] != list(range(120)):
        raise RuntimeError(f"{run}: window sequence")

    if any(r["enforcement_enabled"] != spec["enforcement"] for r in events):
        raise RuntimeError(f"{run}: enforcement mismatch")

    labels = [int(r["predicted_label"]) for r in events]

    tn = fp = fn = tp = 0
    for pred in labels:
        if spec["true_label"] == 0 and pred == 0:
            tn += 1
        elif spec["true_label"] == 0 and pred == 1:
            fp += 1
        elif spec["true_label"] == 1 and pred == 0:
            fn += 1
        else:
            tp += 1

    overall["tn"] += tn
    overall["fp"] += fp
    overall["fn"] += fn
    overall["tp"] += tp

    c = condition_counts.setdefault(
        spec["condition"],
        {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    )

    c["tn"] += tn
    c["fp"] += fp
    c["fn"] += fn
    c["tp"] += tp

    status_client = Counter(r["status_code"] for r in client)
    status_victim = Counter(r["response_status"] for r in victim)

    if status_client != status_victim:
        raise RuntimeError(f"{run}: client/victim status mismatch")

    transport_success = sum(
        r["transport_success"].lower() == "true"
        for r in client
    )

    activated = [
        r for r in events
        if r["enforcement_outcome"] == "activated"
    ]

    post_requests = None
    post_429 = None
    post_failures = None

    if run.startswith("M"):
        if len(activated) != 1:
            raise RuntimeError(f"{run}: activation count")

        a = activated[0]

        if a["predicted_label"] != "1":
            raise RuntimeError(f"{run}: activation prediction")

        if a["mitigation_action_requested"].lower() != "true":
            raise RuntimeError(f"{run}: activation action")

        if not a["enforcement_completion_timestamp"]:
            raise RuntimeError(f"{run}: enforcement timestamp")

        completion = dt(a["enforcement_completion_timestamp"])

        after = [
            r for r in victim
            if dt(r["timestamp_utc"]) >= completion
        ]

        post_requests = len(after)
        post_429 = sum(r["response_status"] == "429" for r in after)
        post_failures = sum(
            r["response_status"] != "429"
            or r["mitigation_active"].lower() != "true"
            for r in after
        )

    if summary["processing_integrity_pass"] is not True:
        raise RuntimeError(f"{run}: processing integrity")

    if summary["accounting_identity_pass"] is not True:
        raise RuntimeError(f"{run}: accounting identity")

    if summary["complete_windows"] != 120:
        raise RuntimeError(f"{run}: complete windows")

    if summary["eligible_tail_packets"] != 0:
        raise RuntimeError(f"{run}: tail")

    raw[run] = {
        "condition": spec["condition"],
        "true_label": spec["true_label"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "client_status_counts": dict(status_client),
        "transport_successes": transport_success,
        "activation_events": len(activated),
        "post_enforcement_requests": post_requests,
        "post_enforcement_http_429": post_429,
        "post_enforcement_failures": post_failures,
        "final_mitigation_active": state["active"],
    }

derived_overall = cm_metrics(
    overall["tn"],
    overall["fp"],
    overall["fn"],
    overall["tp"],
)

derived_conditions = {}

for condition, x in condition_counts.items():
    derived_conditions[condition] = cm_metrics(
        x["tn"], x["fp"], x["fn"], x["tp"]
    )

s6 = json.loads(S6.read_text())
s7 = json.loads(S7.read_text())

checks = {}

checks["S6_OVERALL_CONFUSION_MATRIX"] = (
    s6["pooled_overall"]["confusion_matrix"]
    == derived_overall["confusion_matrix"]
)

for key in [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
]:
    checks[f"S6_OVERALL_{key.upper()}"] = (
        s6["pooled_overall"][key] == derived_overall[key]
    )

for condition, derived in derived_conditions.items():
    reported = s6["pooled_within_condition"][condition]

    checks[f"S6_{condition}_CM"] = (
        reported["confusion_matrix"]
        == derived["confusion_matrix"]
    )

    checks[f"S6_{condition}_ACCURACY"] = (
        reported["accuracy"] == derived["accuracy"]
    )

b200 = sum(
    raw[r]["client_status_counts"].get("200", 0)
    for r in ["B01", "B02", "B03"]
)

a200 = sum(
    raw[r]["client_status_counts"].get("200", 0)
    for r in ["A01", "A02", "A03"]
)

m200 = sum(
    raw[r]["client_status_counts"].get("200", 0)
    for r in ["M01", "M02", "M03"]
)

m429 = sum(
    raw[r]["client_status_counts"].get("429", 0)
    for r in ["M01", "M02", "M03"]
)

post_requests = sum(
    raw[r]["post_enforcement_requests"]
    for r in ["M01", "M02", "M03"]
)

post_429 = sum(
    raw[r]["post_enforcement_http_429"]
    for r in ["M01", "M02", "M03"]
)

post_failures = sum(
    raw[r]["post_enforcement_failures"]
    for r in ["M01", "M02", "M03"]
)

checks["S7_B_HTTP_200"] = (
    s7["by_condition"]["benign_with_detection_and_mitigation"]
    ["client_http_status_counts"].get("200", 0)
    == b200
)

checks["S7_A_HTTP_200"] = (
    s7["by_condition"]["attack_detection_without_mitigation"]
    ["client_http_status_counts"].get("200", 0)
    == a200
)

checks["S7_M_HTTP_200"] = (
    s7["by_condition"]["attack_detection_with_mitigation"]
    ["client_http_status_counts"].get("200", 0)
    == m200
)

checks["S7_M_HTTP_429"] = (
    s7["by_condition"]["attack_detection_with_mitigation"]
    ["client_http_status_counts"].get("429", 0)
    == m429
)

checks["S7_POST_ENFORCEMENT_REQUESTS"] = (
    s7["mitigation_summary"]["post_enforcement_targeted_requests"]
    == post_requests
)

checks["S7_POST_ENFORCEMENT_HTTP_429"] = (
    s7["mitigation_summary"]["post_enforcement_http_429"]
    == post_429
)

checks["S7_POST_ENFORCEMENT_FAILURES"] = (
    s7["mitigation_summary"]["post_enforcement_failures"]
    == post_failures
)

checks["ALL_TRANSPORT_SUCCESS"] = (
    sum(x["transport_successes"] for x in raw.values()) == 9000
)

checks["M_ACTIVATION_COUNT"] = (
    sum(raw[r]["activation_events"] for r in ["M01", "M02", "M03"]) == 3
)

checks["B_NO_ACTIVATION"] = (
    sum(raw[r]["activation_events"] for r in ["B01", "B02", "B03"]) == 0
)

checks["A_NO_ACTIVATION"] = (
    sum(raw[r]["activation_events"] for r in ["A01", "A02", "A03"]) == 0
)

checks["M_FINAL_STATE_ACTIVE"] = all(
    raw[r]["final_mitigation_active"]
    for r in ["M01", "M02", "M03"]
)

checks["B_FINAL_STATE_INACTIVE"] = all(
    not raw[r]["final_mitigation_active"]
    for r in ["B01", "B02", "B03"]
)

checks["A_FINAL_STATE_INACTIVE"] = all(
    not raw[r]["final_mitigation_active"]
    for r in ["A01", "A02", "A03"]
)

result = {
    "experiment": "Exp011",
    "audit": "independent_cross_artifact_scientific_reconciliation",
    "raw_run_derivations": raw,
    "derived_detection_overall": derived_overall,
    "derived_detection_by_condition": derived_conditions,
    "derived_service": {
        "benign_http_200": b200,
        "attack_without_mitigation_http_200": a200,
        "attack_with_mitigation_http_200": m200,
        "attack_with_mitigation_http_429": m429,
        "post_enforcement_requests": post_requests,
        "post_enforcement_http_429": post_429,
        "post_enforcement_failures": post_failures,
    },
    "reconciliation_checks": checks,
    "all_checks_pass": all(checks.values()),
}

out = ROOT / "forensic_closure_v1"
out.mkdir(parents=True, exist_ok=True)

(out / "scientific_reconciliation.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)

for key, value in checks.items():
    print(key, value)

print()
print("DERIVED_OVERALL", derived_overall)
print("B_HTTP_200", b200)
print("A_HTTP_200", a200)
print("M_HTTP_200", m200)
print("M_HTTP_429", m429)
print("POST_ENFORCEMENT_REQUESTS", post_requests)
print("POST_ENFORCEMENT_HTTP_429", post_429)
print("POST_ENFORCEMENT_FAILURES", post_failures)
print()
print("ALL_RECONCILIATION_CHECKS_PASS", all(checks.values()))
