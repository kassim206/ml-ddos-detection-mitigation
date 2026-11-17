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
EVAL_ROOT = ROOT / "evaluation_v1"
INPUT_MANIFEST = EVAL_ROOT / "input_manifest.json"

RUNS = {
    "B01": ("benign_with_detection_and_mitigation", 0),
    "B02": ("benign_with_detection_and_mitigation", 0),
    "B03": ("benign_with_detection_and_mitigation", 0),
    "A01": ("attack_detection_without_mitigation", 1),
    "A02": ("attack_detection_without_mitigation", 1),
    "A03": ("attack_detection_without_mitigation", 1),
    "M01": ("attack_detection_with_mitigation", 1),
    "M02": ("attack_detection_with_mitigation", 1),
    "M03": ("attack_detection_with_mitigation", 1),
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

def finite(x):
    return math.isfinite(x)

def safe_div(a, b):
    return a / b if b else None

def metrics(y_true, y_pred):
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))

    accuracy = safe_div(tp + tn, len(y_true))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    f1 = None
    if precision is not None and recall is not None:
        f1 = safe_div(2 * precision * recall, precision + recall)
        if precision == 0 and recall == 0:
            f1 = 0.0

    balanced_accuracy = None
    if tn + fp and tp + fn:
        balanced_accuracy = (
            tn / (tn + fp) + tp / (tp + fn)
        ) / 2

    p0 = safe_div(tn, tn + fn)
    r0 = safe_div(tn, tn + fp)
    if p0 is None or r0 is None or p0 + r0 == 0:
        f10 = 0.0
    else:
        f10 = 2 * p0 * r0 / (p0 + r0)

    p1 = safe_div(tp, tp + fp)
    r1 = safe_div(tp, tp + fn)
    if p1 is None or r1 is None or p1 + r1 == 0:
        f11 = 0.0
    else:
        f11 = 2 * p1 * r1 / (p1 + r1)

    macro_f1 = (f10 + f11) / 2

    return {
        "n_windows": len(y_true),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }

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

EVAL_ROOT.mkdir(parents=True, exist_ok=True)

input_manifest = json.loads(INPUT_MANIFEST.read_text())

if set(input_manifest["result_runs"]) != set(RUNS):
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

    current_manifest_hash = sha256(p / "SHA256SUMS")
    expected_manifest_hash = input_manifest["result_runs"][run]["manifest_sha256"]

    integrity[run] = {
        "artifact_manifest_verified": r1.returncode == 0,
        "manifest_hash_verified": r2.returncode == 0,
        "input_manifest_binding_verified": current_manifest_hash == expected_manifest_hash,
        "manifest_sha256": current_manifest_hash,
    }

    if not all([
        integrity[run]["artifact_manifest_verified"],
        integrity[run]["manifest_hash_verified"],
        integrity[run]["input_manifest_binding_verified"],
    ]):
        raise RuntimeError(f"integrity verification failed for {run}")

per_run = {}
all_records = []
timing_records = []

for run, (condition, true_label) in RUNS.items():
    events = read_csv(RUN_ROOT / run / "detector_events.csv")

    if len(events) != 120:
        raise RuntimeError(f"{run}: expected 120 windows")

    indices = [int(r["window_index"]) for r in events]
    if indices != list(range(120)):
        raise RuntimeError(f"{run}: window indices invalid")

    y_true = [true_label] * len(events)
    y_pred = [int(r["predicted_label"]) for r in events]

    if any(x not in (0, 1) for x in y_pred):
        raise RuntimeError(f"{run}: invalid predicted label")

    probabilities = [float(r["attack_probability"]) for r in events]

    if not all(finite(x) for x in probabilities):
        raise RuntimeError(f"{run}: non-finite probability")

    window_observation = []
    feature_to_inference = []
    inference = []
    decision_to_enforcement = []

    for r in events:
        wo = (
            dt(r["window_end_timestamp"])
            - dt(r["window_start_timestamp"])
        ).total_seconds()

        fi = (
            dt(r["inference_completion_timestamp"])
            - dt(r["feature_completion_timestamp"])
        ).total_seconds()

        inf = (
            dt(r["inference_completion_timestamp"])
            - dt(r["inference_start_timestamp"])
        ).total_seconds()

        if wo < 0 or fi < 0 or inf < 0:
            raise RuntimeError(f"{run}: negative frozen timing quantity")

        window_observation.append(wo)
        feature_to_inference.append(fi)
        inference.append(inf)

        de = None

        if r["enforcement_completion_timestamp"]:
            de = (
                dt(r["enforcement_completion_timestamp"])
                - dt(r["attack_decision_timestamp"])
            ).total_seconds()

            if de < 0:
                raise RuntimeError(f"{run}: negative decision-to-enforcement")

            decision_to_enforcement.append(de)

        all_records.append({
            "run_id": run,
            "condition": condition,
            "true_label": true_label,
            "predicted_label": int(r["predicted_label"]),
        })

        timing_records.append({
            "run_id": run,
            "condition": condition,
            "window_index": int(r["window_index"]),
            "window_observation_seconds": wo,
            "feature_to_inference_seconds": fi,
            "inference_seconds": inf,
            "decision_to_enforcement_seconds": de,
        })

    per_run[run] = {
        "condition": condition,
        "true_label": true_label,
        "detection": metrics(y_true, y_pred),
        "attack_probability": describe(probabilities),
        "timing": {
            "window_observation_seconds": describe(window_observation),
            "feature_to_inference_seconds": describe(feature_to_inference),
            "inference_seconds": describe(inference),
            "decision_to_enforcement_seconds": describe(decision_to_enforcement),
        },
    }

pooled_condition = {}

for condition in CONDITIONS:
    subset = [r for r in all_records if r["condition"] == condition]
    pooled_condition[condition] = metrics(
        [r["true_label"] for r in subset],
        [r["predicted_label"] for r in subset],
    )

overall = metrics(
    [r["true_label"] for r in all_records],
    [r["predicted_label"] for r in all_records],
)

timing_by_condition = {}

for condition in CONDITIONS:
    subset = [r for r in timing_records if r["condition"] == condition]

    timing_by_condition[condition] = {
        "window_observation_seconds": describe([
            r["window_observation_seconds"] for r in subset
        ]),
        "feature_to_inference_seconds": describe([
            r["feature_to_inference_seconds"] for r in subset
        ]),
        "inference_seconds": describe([
            r["inference_seconds"] for r in subset
        ]),
        "decision_to_enforcement_seconds": describe([
            r["decision_to_enforcement_seconds"]
            for r in subset
            if r["decision_to_enforcement_seconds"] is not None
        ]),
    }

result = {
    "experiment": "Exp011",
    "evaluation_version": 1,
    "input_manifest_sha256": sha256(INPUT_MANIFEST),
    "experimental_unit": "run",
    "detection_observation_unit": "complete non-overlapping 100-eligible-packet window",
    "replication_warning": "Packet windows are nested observations within runs and are not independent experimental replicates.",
    "ground_truth_rule": {
        "B01-B03": 0,
        "A01-A03": 1,
        "M01-M03": 1,
    },
    "integrity": integrity,
    "per_run": per_run,
    "pooled_within_condition": pooled_condition,
    "pooled_overall": overall,
    "timing_by_condition": timing_by_condition,
    "timing_scope_warning": "Timing quantities retain the frozen metric definitions and are not end-to-end request latency.",
}

(EVAL_ROOT / "detection_runtime_results.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)

with (EVAL_ROOT / "detection_per_run.csv").open("w", newline="") as f:
    fields = [
        "run_id",
        "condition",
        "true_label",
        "n_windows",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
    ]

    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    for run in RUNS:
        d = per_run[run]["detection"]
        cm = d["confusion_matrix"]

        w.writerow({
            "run_id": run,
            "condition": per_run[run]["condition"],
            "true_label": per_run[run]["true_label"],
            "n_windows": d["n_windows"],
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
            "accuracy": d["accuracy"],
            "balanced_accuracy": d["balanced_accuracy"],
            "precision": d["precision"],
            "recall": d["recall"],
            "f1": d["f1"],
            "macro_f1": d["macro_f1"],
        })

print("INPUT_MANIFEST_SHA256", sha256(INPUT_MANIFEST))
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
    d = per_run[run]["detection"]
    print(
        run,
        d["confusion_matrix"],
        "ACC", d["accuracy"],
        "BA", d["balanced_accuracy"],
        "PREC", d["precision"],
        "REC", d["recall"],
        "F1", d["f1"],
        "MACRO_F1", d["macro_f1"],
    )

print()
print("POOLED_WITHIN_CONDITION")

for condition in CONDITIONS:
    print(condition, pooled_condition[condition])

print()
print("POOLED_OVERALL", overall)

print()
print("TIMING_BY_CONDITION")

for condition in CONDITIONS:
    print(condition)
    for metric, values in timing_by_condition[condition].items():
        print(metric, values)

print()
print(
    "RESULT_SHA256",
    sha256(EVAL_ROOT / "detection_runtime_results.json")
)
print(
    "PER_RUN_CSV_SHA256",
    sha256(EVAL_ROOT / "detection_per_run.csv")
)
