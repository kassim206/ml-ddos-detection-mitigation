#!/usr/bin/env python3

import csv
import json
from pathlib import Path

MATRIX = Path(
    "runs/thesis_evidence_reconciliation_v1/"
    "thesis_evidence_matrix.json"
)

OUT = Path("tables")


def load(path):
    return json.loads(Path(path).read_text())


def write_csv(name, fields, rows):
    path = OUT / name
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


m = load(MATRIX)
by = {x["experiment"]: x for x in m["experiments"]}

expected_experiments = {
    "Exp001",
    "Exp002",
    "Exp003",
    "Exp004",
    "Exp005",
    "Exp006",
    "Exp007",
    "Exp008",
    "Exp009",
    "Exp010",
    "Exp011"
}

if set(by) != expected_experiments:
    raise RuntimeError(
        f"Unexpected experiment set: {sorted(by)}"
    )

OUT.mkdir(parents=True, exist_ok=True)

for p in OUT.glob("table_*.csv"):
    p.unlink()

e1 = by["Exp001"]["evidence"]
e2 = by["Exp002"]["evidence"]

write_csv(
    "table_01_dataset_a_composition_split.csv",
    [
        "item",
        "value"
    ],
    [
        {
            "item": "Source feature CSV files",
            "value": e1["source_files"]
        },
        {
            "item": "Total rows",
            "value": e1["total_rows"]
        },
        {
            "item": "Training rows",
            "value": e1["train_rows"]
        },
        {
            "item": "Validation rows",
            "value": e1["validation_rows"]
        },
        {
            "item": "Test rows",
            "value": e1["test_rows"]
        },
        {
            "item": "Feature count",
            "value": e2["feature_count"]
        },
        {
            "item": "Whole-file allocations",
            "value": e1["whole_file_count"]
        },
        {
            "item": "Row-level partitioned files",
            "value": e1["row_level_partitioned_file_count"]
        }
    ]
)

e3 = by["Exp003"]["evidence"]
e4 = by["Exp004"]["evidence"]
e9 = by["Exp009"]["evidence"]

write_csv(
    "table_02_dataset_a_model_performance.csv",
    [
        "experiment",
        "model",
        "evaluation_scope",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1"
    ],
    [
        {
            "experiment": "Exp003",
            "model": "Random Forest",
            "evaluation_scope": "held-out test",
            "accuracy": e3["RandomForest"]["accuracy"],
            "macro_precision":
                e3["RandomForest"]["macro_precision"],
            "macro_recall":
                e3["RandomForest"]["macro_recall"],
            "macro_f1":
                e3["RandomForest"]["macro_f1"],
            "weighted_f1": ""
        },
        {
            "experiment": "Exp003",
            "model": "XGBoost",
            "evaluation_scope": "held-out test",
            "accuracy": e3["XGBoost"]["accuracy"],
            "macro_precision":
                e3["XGBoost"]["macro_precision"],
            "macro_recall":
                e3["XGBoost"]["macro_recall"],
            "macro_f1":
                e3["XGBoost"]["macro_f1"],
            "weighted_f1": ""
        },
        {
            "experiment": "Exp004",
            "model": "CNN-Dense",
            "evaluation_scope": "validation only",
            "accuracy": e4["accuracy"],
            "macro_precision": "",
            "macro_recall": "",
            "macro_f1": e4["macro_f1"],
            "weighted_f1": e4["weighted_f1"]
        },
        {
            "experiment": "Exp009",
            "model": "Hybrid CNN-LSTM",
            "evaluation_scope": "held-out test",
            "accuracy": e9["accuracy"],
            "macro_precision": e9["macro_precision"],
            "macro_recall": e9["macro_recall"],
            "macro_f1": e9["macro_f1"],
            "weighted_f1": e9["weighted_f1"]
        }
    ]
)

e5 = by["Exp005"]["evidence"]

write_csv(
    "table_03_attack_family_performance.csv",
    [
        "attack_family",
        "broad_category",
        "support",
        "category_recall",
        "mean_confidence"
    ],
    [
        {
            "attack_family": name,
            "broad_category": x["category"],
            "support": x["support"],
            "category_recall": x["category_recall"],
            "mean_confidence": x["mean_confidence"]
        }
        for name, x in e5.items()
    ]
)

e6 = by["Exp006"]["evidence"]

write_csv(
    "table_04_cpu_batch_inference_runtime.csv",
    [
        "model",
        "repeats",
        "rows_per_repeat",
        "mean_total_seconds",
        "std_total_seconds",
        "mean_amortized_ms_per_sample",
        "throughput_samples_per_second"
    ],
    [
        {
            "model": model,
            "repeats": x["repeats"],
            "rows_per_repeat": x["rows_per_repeat"],
            "mean_total_seconds": x["mean_total_seconds"],
            "std_total_seconds": x["std_total_seconds"],
            "mean_amortized_ms_per_sample":
                x["mean_amortized_ms_per_sample"],
            "throughput_samples_per_second":
                x["throughput_samples_per_second"]
        }
        for model, x in e6.items()
    ]
)

e7 = by["Exp007"]["evidence"]

write_csv(
    "table_05_controlled_http_service_baseline.csv",
    [
        "metric",
        "value"
    ],
    [
        {
            "metric": "Requests attempted",
            "value": e7["requests_attempted"]
        },
        {
            "metric": "Transport successful requests",
            "value": e7["transport_successful_requests"]
        },
        {
            "metric": "Total elapsed seconds",
            "value": e7["total_elapsed_seconds"]
        },
        {
            "metric": "Throughput requests per second",
            "value": e7["throughput_requests_per_second"]
        },
        {
            "metric": "Mean response time ms",
            "value": e7["mean_response_time_ms"]
        },
        {
            "metric": "P95 response time ms",
            "value": e7["p95_response_time_ms"]
        }
    ]
)

e8 = by["Exp008"]["evidence"]

write_csv(
    "table_06_offline_mitigation_policy.csv",
    [
        "metric",
        "value"
    ],
    [
        {
            "metric": "Decision threshold",
            "value": e8["threshold"]
        },
        {
            "metric": "Total observations",
            "value": e8["total_observations"]
        },
        {
            "metric": "Malicious block rate",
            "value": e8["malicious_block_rate"]
        },
        {
            "metric": "Benign block rate",
            "value": e8["benign_block_rate"]
        },
        {
            "metric": "DDoS block rate",
            "value": e8["DDoS_block_rate"]
        },
        {
            "metric": "Recon block rate",
            "value": e8["Recon_block_rate"]
        },
        {
            "metric": "Web block rate",
            "value": e8["Web_block_rate"]
        }
    ]
)

e10 = by["Exp010"]["evidence"]
pm = e10["pooled_metrics"]

write_csv(
    "table_07_dataset_b_heldout_detection.csv",
    [
        "metric",
        "value"
    ],
    [
        {
            "metric": "Selected model",
            "value": e10["selected_model"]
        },
        {
            "metric": "Threshold",
            "value": e10["threshold"]
        },
        {
            "metric": "Window size packets",
            "value": e10["window_size_packets"]
        },
        {
            "metric": "Held-out windows",
            "value": e10["total_test_windows"]
        },
        {
            "metric": "Accuracy",
            "value": pm["accuracy"]
        },
        {
            "metric": "Balanced accuracy",
            "value": pm["balanced_accuracy"]
        },
        {
            "metric": "Macro-F1",
            "value": pm["macro_f1"]
        },
        {
            "metric": "Confusion matrix",
            "value": json.dumps(pm["confusion_matrix"])
        }
    ]
)

e11 = by["Exp011"]["evidence"]
pd = e11["pooled_detection"]
timing = e11["timing_by_condition"]

table_08_fields = [
    "scope",
    "condition",
    "experimental_unit",
    "observed_windows",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "window_observation_mean_seconds",
    "feature_to_inference_mean_seconds",
    "inference_mean_seconds",
    "decision_to_enforcement_mean_seconds"
]

table_08_rows = [
    {
        "scope": "pooled_overall",
        "condition": "all_conditions",
        "experimental_unit": e11["experimental_unit"],
        "observed_windows": pd["n_windows"],
        "accuracy": pd["accuracy"],
        "balanced_accuracy": pd["balanced_accuracy"],
        "precision": pd["precision"],
        "recall": pd["recall"],
        "f1": pd["f1"],
        "macro_f1": pd["macro_f1"],
        "window_observation_mean_seconds": "",
        "feature_to_inference_mean_seconds": "",
        "inference_mean_seconds": "",
        "decision_to_enforcement_mean_seconds": ""
    }
]

for condition, x in timing.items():
    table_08_rows.append(
        {
            "scope": "condition_timing",
            "condition": condition,
            "experimental_unit": e11["experimental_unit"],
            "observed_windows": 360,
            "accuracy": "",
            "balanced_accuracy": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "macro_f1": "",
            "window_observation_mean_seconds":
                x["window_observation_seconds"]["mean"],
            "feature_to_inference_mean_seconds":
                x["feature_to_inference_seconds"]["mean"],
            "inference_mean_seconds":
                x["inference_seconds"]["mean"],
            "decision_to_enforcement_mean_seconds":
                x["decision_to_enforcement_seconds"]["mean"]
        }
    )

write_csv(
    "table_08_live_detection_runtime.csv",
    table_08_fields,
    table_08_rows
)

service = e11["service_by_condition"]
mitigation = e11["mitigation_summary"]

table_09_rows = []

for condition, x in service.items():
    row = {
        "condition": condition,
        "runs": len(x["runs"]),
        "request_attempts": x["request_attempts"],
        "successful_transport_requests":
            x["successful_transport_requests"],
        "failed_or_timed_out_requests":
            x["failed_or_timed_out_requests"],
        "http_200":
            x["client_http_status_counts"].get("200", 0),
        "http_429":
            x["client_http_status_counts"].get("429", 0),
        "mean_request_span_seconds":
            x["request_span_seconds_per_run"]["mean"],
        "mean_request_span_throughput":
            x["request_span_throughput_per_run"]["mean"],
        "mean_request_latency_ms":
            x["latency_mean_ms_per_run"]["mean"],
        "post_enforcement_targeted_requests": "",
        "post_enforcement_http_429": "",
        "post_enforcement_failures": "",
        "decision_to_enforcement_mean_seconds": ""
    }

    if condition == "attack_detection_with_mitigation":
        row["post_enforcement_targeted_requests"] = (
            mitigation["post_enforcement_targeted_requests"]
        )
        row["post_enforcement_http_429"] = (
            mitigation["post_enforcement_http_429"]
        )
        row["post_enforcement_failures"] = (
            mitigation["post_enforcement_failures"]
        )
        row["decision_to_enforcement_mean_seconds"] = (
            mitigation["decision_to_enforcement_seconds"]["mean"]
        )

    table_09_rows.append(row)

write_csv(
    "table_09_live_mitigation_service_impact.csv",
    [
        "condition",
        "runs",
        "request_attempts",
        "successful_transport_requests",
        "failed_or_timed_out_requests",
        "http_200",
        "http_429",
        "mean_request_span_seconds",
        "mean_request_span_throughput",
        "mean_request_latency_ms",
        "post_enforcement_targeted_requests",
        "post_enforcement_http_429",
        "post_enforcement_failures",
        "decision_to_enforcement_mean_seconds"
    ],
    table_09_rows
)

expected_tables = [
    "table_01_dataset_a_composition_split.csv",
    "table_02_dataset_a_model_performance.csv",
    "table_03_attack_family_performance.csv",
    "table_04_cpu_batch_inference_runtime.csv",
    "table_05_controlled_http_service_baseline.csv",
    "table_06_offline_mitigation_policy.csv",
    "table_07_dataset_b_heldout_detection.csv",
    "table_08_live_detection_runtime.csv",
    "table_09_live_mitigation_service_impact.csv"
]

actual_tables = sorted(
    p.name
    for p in OUT.glob("table_*.csv")
)

if actual_tables != expected_tables:
    raise RuntimeError(
        f"Unexpected final table set: {actual_tables}"
    )

if e1["total_rows"] != (
    e1["train_rows"]
    + e1["validation_rows"]
    + e1["test_rows"]
):
    raise RuntimeError("Dataset A split rows do not reconcile")

if pd["n_windows"] != 1080:
    raise RuntimeError("Unexpected Exp011 pooled window count")

if len(timing) != 3:
    raise RuntimeError("Unexpected Exp011 timing condition count")

if sum(
    x["request_attempts"]
    for x in service.values()
) != 9000:
    raise RuntimeError("Unexpected Exp011 request total")

if (
    mitigation["post_enforcement_targeted_requests"]
    != mitigation["post_enforcement_http_429"]
):
    raise RuntimeError(
        "Post-enforcement request and HTTP 429 counts differ"
    )

if mitigation["post_enforcement_failures"] != 0:
    raise RuntimeError(
        "Unexpected post-enforcement mitigation failures"
    )

print("FINAL_THESIS_TABLES", len(actual_tables))

for name in actual_tables:
    print(OUT / name)