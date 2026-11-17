#!/usr/bin/env python3

import csv
import hashlib
import json
from pathlib import Path

R = Path("runs")
OUT = R / "thesis_evidence_reconciliation_v1"

def load(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def source(path):
    p = Path(path)
    return {
        "path": str(p),
        "sha256": sha(p)
    }

e1_summary_path = R / "experiment_001_dataset_char/experiment_summary.json"
e1_dataset_path = R / "experiment_001_dataset_char/dataset_manifest.json"
e1_split_path = R / "experiment_001_dataset_char/split_manifest.json"
e2_config_path = R / "experiment_002_preprocessing/config.json"
e2_log_path = R / "experiment_002_preprocessing/preprocessing_log.json"
e3_rf_path = R / "experiment_003_baselines/metrics_rf.json"
e3_xgb_path = R / "experiment_003_baselines/metrics_xgb.json"
e4_path = R / "experiment_004_cnn_dense/diagnostics_best_checkpoint/validation_metrics_best_checkpoint.json"
e5_path = R / "experiment_005_per_attack/attack_family_metrics.json"
e6_config_path = R / "experiment_006_runtime/config.json"
e6_runtime_path = R / "experiment_006_runtime/resource_usage.csv"
e7_path = R / "experiment_007_web_testbed/summary.json"
e8_summary_path = R / "experiment_008_mitigation/experiment_summary.json"
e8_metrics_path = R / "experiment_008_mitigation/mitigation_metrics.json"
e9_log_path = R / "experiment_009_hybrid_cnn_lstm/evaluation_log.json"
e9_metrics_path = R / "experiment_009_hybrid_cnn_lstm/test_metrics.json"
e10_path = R / "experiment_010_realtime_detection_mitigation/deployment_model/final_test_evaluation_v1/final_test_results.json"
e10_selected_path = R / "experiment_010_realtime_detection_mitigation/deployment_model/final_test_evaluation_v1/selected_system.json"
e11_detection_path = R / "experiment_011_live_detection_mitigation/evaluation_v1/detection_runtime_results.json"
e11_service_path = R / "experiment_011_live_detection_mitigation/service_mitigation_evaluation_v1/mitigation_service_results.json"
e11_closure_path = R / "experiment_011_live_detection_mitigation/forensic_closure_v1/section8_status.json"

e1 = load(e1_summary_path)
e1d = load(e1_dataset_path)
e1s = load(e1_split_path)
e2c = load(e2_config_path)
e2 = load(e2_log_path)
rf = load(e3_rf_path)
xgb = load(e3_xgb_path)
e4 = load(e4_path)
e5 = load(e5_path)
e6c = load(e6_config_path)
e7 = load(e7_path)
e8s = load(e8_summary_path)
e8 = load(e8_metrics_path)
e9l = load(e9_log_path)
e9 = load(e9_metrics_path)
e10 = load(e10_path)
e10s = load(e10_selected_path)
e11d = load(e11_detection_path)
e11s = load(e11_service_path)
e11c = load(e11_closure_path)

with e6_runtime_path.open(newline="") as f:
    e6_rows = list(csv.DictReader(f))

runtime = {
    r["model"]: {
        "mean_total_seconds": float(r["mean_total_seconds"]),
        "std_total_seconds": float(r["std_total_seconds"]),
        "mean_amortized_ms_per_sample": float(r["mean_amortized_ms_per_sample"]),
        "throughput_samples_per_second": float(r["throughput_samples_per_second"]),
        "repeats": int(r["repeats"]),
        "rows_per_repeat": int(r["rows_per_repeat"])
    }
    for r in e6_rows
}

attack_metrics = {
    x["attack_family"]: x
    for x in e5["attack_family_metrics"]
}

checks = {
    "dataset_total_consistent": e1d["total_rows"] == e1s["total_rows"] == e2["total_rows"],
    "dataset_file_count_consistent": e1d["file_count"] == e1s["total_files"] == 46,
    "exp001_deterministic": e1d["deterministic"] is True and e1s["deterministic"] is True,
    "exp002_feature_count_39": e2c["feature_count"] == e2["feature_count"] == 39,
    "exp002_train_only_scaler": e2["scaler_fit_source"] == "training_split_only",
    "exp002_rows_preserved": e2["row_count_verified"] is True and e2["data_loss_detected"] is False,
    "exp003_same_test_rows": rf["test_rows"] == xgb["test_rows"] == e2["test_rows"],
    "exp004_is_cnn_dense": e4["model_name"] == "CNN_Dense",
    "exp004_validation_only_source": e4["validation_file"].endswith("val_processed.parquet"),
    "exp005_alignment_verified": e5["row_alignment_verified"] is True,
    "exp006_full_test": e6c["all_test_rows_processed"] is True,
    "exp006_three_repeats": e6c["measured_repeats"] == 3,
    "exp007_not_hardcoded": e7["hardcoded_measurements"] is False,
    "exp008_offline": e8s["live_feature_extraction_claimed"] is False and e8s["live_model_inference_claimed"] is False and e8s["live_enforcement_claimed"] is False,
    "exp009_feature_count_39": e9l["feature_count"] == 39,
    "exp009_test_rows_consistent": e9["rows_evaluated"] == e2["test_rows"],
    "exp010_dataset_b": e10["dataset"] == "Dataset B",
    "exp010_test_evaluated_once": e10["status"] == "FINAL_TEST_EVALUATED_ONCE",
    "exp010_test_was_untouched_at_selection": e10s["test_runs_accessed"] is False,
    "exp011_experimental_unit_run": e11d["experimental_unit"] == "run" and e11s["experimental_unit"] == "run",
    "exp011_forensic_gate": e11c["forensic_manifest_gate_pass"] is True,
    "exp011_reconciliation": e11c["scientific_reconciliation_pass"] is True,
    "exp011_wall_clock_not_supported": e11s["service_metric_qualification"]["wall_clock_seconds"] == "NOT_SUPPORTED_FROM_ARCHIVED_ARTIFACTS"
}

if not all(checks.values()):
    failed = [k for k, v in checks.items() if not v]
    raise RuntimeError(f"Cross-experiment invariants failed: {failed}")

experiments = [
    {
        "experiment": "Exp001",
        "role": "Dataset A characterization and deterministic splitting",
        "status": "PASS WITH QUALIFICATION",
        "sources": [source(e1_summary_path), source(e1_dataset_path), source(e1_split_path)],
        "evidence": {
            "dataset": e1d["dataset"],
            "source_files": e1d["file_count"],
            "total_rows": e1d["total_rows"],
            "split_method": e1d["split_method"],
            "deterministic": e1d["deterministic"],
            "random_seed": e1d["random_seed"],
            "train_rows": e1["actual_rows"]["train"],
            "validation_rows": e1["actual_rows"]["val"],
            "test_rows": e1["actual_rows"]["test"],
            "whole_file_count": e1["whole_file_count"],
            "row_level_partitioned_file_count": len(e1["row_level_partitioned_files"])
        },
        "permitted_interpretation": "Selected CICIoT2023 feature-CSV subset split deterministically with whole-file allocation preferred and row-level partitioning where required.",
        "required_qualification": "This is a selected subset rather than the full CICIoT2023 corpus; the split is not a temporal split and is not purely file-level.",
        "prohibited_claims": [
            "Full CICIoT2023 was used",
            "The split was purely file-level",
            "The split guarantees temporal independence",
            "The .pcap.csv inputs are raw PCAP captures"
        ]
    },
    {
        "experiment": "Exp002",
        "role": "Dataset A preprocessing",
        "status": "PASS",
        "sources": [source(e2_config_path), source(e2_log_path)],
        "evidence": {
            "feature_count": e2["feature_count"],
            "train_rows": e2["train_rows"],
            "validation_rows": e2["validation_rows"],
            "test_rows": e2["test_rows"],
            "scaler_fit_source": e2["scaler_fit_source"],
            "scaling_method": e2["scaling_method"],
            "nan_handling": e2["nan_handling"],
            "row_count_verified": e2["row_count_verified"]
        },
        "permitted_interpretation": "Thirty-nine features were standardized using a StandardScaler fitted only on the training split, with non-finite feature values replaced by zero before scaling.",
        "required_qualification": "The preprocessing contract contains 39 features, not 44.",
        "prohibited_claims": [
            "Forty-four input features were used",
            "The scaler was fitted using validation or test data"
        ]
    },
    {
        "experiment": "Exp003",
        "role": "Dataset A Random Forest and XGBoost baselines",
        "status": "PASS",
        "sources": [source(e3_rf_path), source(e3_xgb_path)],
        "evidence": {
            "RandomForest": {
                "accuracy": rf["accuracy"],
                "macro_precision": rf["precision_macro"],
                "macro_recall": rf["recall_macro"],
                "macro_f1": rf["f1_macro"]
            },
            "XGBoost": {
                "accuracy": xgb["accuracy"],
                "macro_precision": xgb["precision_macro"],
                "macro_recall": xgb["recall_macro"],
                "macro_f1": xgb["f1_macro"]
            }
        },
        "permitted_interpretation": "Both classical baselines achieved high overall accuracy on Dataset A, while substantially lower macro recall and macro-F1 show weaker class-balanced performance.",
        "required_qualification": "Accuracy must be reported with macro metrics because the test distribution is strongly imbalanced.",
        "prohibited_claims": [
            "Approximately 99.5 percent accuracy implies uniformly strong performance across all classes"
        ]
    },
    {
        "experiment": "Exp004",
        "role": "CNN-Dense diagnostic validation",
        "status": "PASS WITH LIMITATION",
        "sources": [source(e4_path)],
        "evidence": {
            "model_name": e4["model_name"],
            "validation_rows": e4["validation_rows"],
            "accuracy": e4["accuracy"],
            "macro_f1": e4["f1_macro"],
            "weighted_f1": e4["f1_weighted"],
            "model_retrained": e4["model_retrained"]
        },
        "permitted_interpretation": "Best-checkpoint CNN-Dense validation diagnostic.",
        "required_qualification": "This is validation-only evidence from an interrupted historical experiment and is not a test-set CNN-LSTM result.",
        "prohibited_claims": [
            "Exp004 is a CNN-LSTM model",
            "Exp004 reports held-out test performance"
        ]
    },
    {
        "experiment": "Exp005",
        "role": "Original attack-family stratification of Exp009 broad-category predictions",
        "status": "PASS",
        "sources": [source(e5_path)],
        "evidence": {
            k: {
                "category": v["category"],
                "support": v["support"],
                "category_recall": v["category_recall"],
                "mean_confidence": v["mean_confidence"]
            }
            for k, v in attack_metrics.items()
        },
        "permitted_interpretation": e5["important_interpretation"],
        "required_qualification": "Zero-support attack families are not evaluated and retain null metrics.",
        "prohibited_claims": [
            "These are eight-class attack-family predictions",
            "Zero-support attack families achieved zero percent recall"
        ]
    },
    {
        "experiment": "Exp006",
        "role": "Dataset A CPU batch-inference runtime benchmark",
        "status": "PASS WITH METHODOLOGICAL QUALIFICATION",
        "sources": [source(e6_config_path), source(e6_runtime_path)],
        "evidence": runtime,
        "permitted_interpretation": "Three-repeat complete-test-split chunked batched CPU inference benchmark.",
        "required_qualification": e6c["latency_interpretation"],
        "prohibited_claims": [
            "The amortized values are independent HTTP request latency",
            "The benchmark measured GPU inference latency",
            "The benchmark measured end-to-end live detection latency"
        ]
    },
    {
        "experiment": "Exp007",
        "role": "Controlled local HTTP service baseline",
        "status": "PASS WITH SCOPE LIMITATION",
        "sources": [source(e7_path)],
        "evidence": {
            "requests_attempted": e7["requests_attempted"],
            "transport_successful_requests": e7["transport_successful_requests"],
            "total_elapsed_seconds": e7["total_elapsed_seconds"],
            "throughput_requests_per_second": e7["observed_wall_clock_throughput_requests_per_second"],
            "mean_response_time_ms": e7["mean_response_time_ms"],
            "p95_response_time_ms": e7["p95_response_time_ms"]
        },
        "permitted_interpretation": "Measured local HTTP workload baseline of 1000 requests.",
        "required_qualification": "This is a service/load baseline and does not itself include ML detection, DDoS generation, or mitigation.",
        "prohibited_claims": [
            "Exp007 demonstrates DDoS detection",
            "Exp007 demonstrates mitigation"
        ]
    },
    {
        "experiment": "Exp008",
        "role": "Offline model-driven mitigation-policy analysis",
        "status": "PASS WITH SCOPE LIMITATION",
        "sources": [source(e8_summary_path), source(e8_metrics_path)],
        "evidence": {
            "threshold": e8["threshold"],
            "total_observations": e8["total_observations"],
            "malicious_block_rate": e8["malicious_block_rate"],
            "benign_block_rate": e8["benign_block_rate"],
            "DDoS_block_rate": e8["DDoS_block_rate"],
            "Recon_block_rate": e8["Recon_block_rate"],
            "Web_block_rate": e8["Web_block_rate"],
            "prediction_source_sha256": e8["prediction_source_sha256"]
        },
        "permitted_interpretation": e8s["scope"],
        "required_qualification": "Policy decisions were evaluated offline from frozen Exp009 predictions; no live enforcement occurred.",
        "prohibited_claims": [
            "Exp008 performed live feature extraction",
            "Exp008 performed live model inference",
            "Exp008 enforced iptables rules",
            "Exp008 demonstrates end-to-end live mitigation"
        ]
    },
    {
        "experiment": "Exp009",
        "role": "Historical Dataset A hybrid CNN-LSTM held-out evaluation",
        "status": "PASS WITH TRAINING-HISTORY LIMITATION",
        "sources": [source(e9_log_path), source(e9_metrics_path)],
        "evidence": {
            "feature_count": e9l["feature_count"],
            "test_rows": e9["rows_evaluated"],
            "accuracy": e9["accuracy"],
            "macro_precision": e9["precision_macro"],
            "macro_recall": e9["recall_macro"],
            "macro_f1": e9["f1_macro"],
            "weighted_f1": e9["f1_weighted"],
            "completed_training_epochs": e9l["completed_training_epochs"],
            "selected_checkpoint_epoch": e9l["selected_checkpoint_epoch"]
        },
        "permitted_interpretation": "Held-out Dataset A evaluation of the preserved best hybrid CNN-LSTM checkpoint.",
        "required_qualification": "The training run was interrupted and the architecture applies convolution/LSTM operations over the feature axis rather than genuine temporal packet sequences.",
        "prohibited_claims": [
            "The hybrid achieved approximately 99 percent held-out accuracy",
            "The hybrid was uniformly strong across classes",
            "The LSTM consumed genuine temporal traffic sequences"
        ]
    },
    {
        "experiment": "Exp010",
        "role": "Deployment-compatible Dataset B model development and held-out controlled-lab test",
        "status": "PASS WITH CONTROLLED-LAB SCOPE",
        "sources": [source(e10_path), source(e10_selected_path)],
        "evidence": {
            "selected_model": e10["selected_model"],
            "threshold": e10["threshold"],
            "window_size_packets": e10["window_size_packets"],
            "total_test_windows": e10["total_test_windows"],
            "pooled_metrics": e10["pooled_metrics"],
            "interpretation_scope": e10["interpretation_scope"]
        },
        "permitted_interpretation": "Held-out controlled-laboratory discrimination between the predefined benign and high-rate HTTP conditions using the frozen deployment-compatible model.",
        "required_qualification": "Dataset B is a separate controlled-lab dataset and must not be presented as semantically equivalent to Dataset A or as broad DDoS generalization.",
        "prohibited_claims": [
            "Exp010 proves perfect general DDoS detection",
            "Dataset B is equivalent to CICIoT2023 Dataset A",
            "The 480 windows are independent experimental replicates"
        ]
    },
    {
        "experiment": "Exp011",
        "role": "End-to-end online detection and actual application-level mitigation",
        "status": e11c["status"],
        "sources": [source(e11_detection_path), source(e11_service_path), source(e11_closure_path)],
        "evidence": {
            "experimental_unit": e11d["experimental_unit"],
            "pooled_detection": e11d["pooled_overall"],
            "timing_by_condition": e11d["timing_by_condition"],
            "mitigation_summary": e11s["mitigation_summary"],
            "service_by_condition": e11s["by_condition"],
            "service_metric_qualification": e11s["service_metric_qualification"]
        },
        "permitted_interpretation": e11c["interpretation"],
        "required_qualification": "The experimental unit is the run; windows and requests are nested. Mitigation is application-level HTTP 429 enforcement in a controlled laboratory.",
        "prohibited_claims": [
            "Exp011 demonstrates production-scale DDoS mitigation",
            "Exp011 used firewall or iptables enforcement",
            "Exp011 performed source-IP blocking",
            "The 1080 windows are independent experimental replicates",
            "Decision-to-enforcement time is end-to-end HTTP request latency",
            "Exact client-process wall-clock duration is recoverable from archived artifacts"
        ]
    }
]

matrix = {
    "artifact": "authoritative_thesis_evidence_matrix",
    "version": 1,
    "experiment_range": "Exp001-Exp011",
    "construction": "Numerical outcomes are extracted from authoritative experiment artifacts. Interpretive boundaries are audit classifications and do not replace underlying experimental evidence.",
    "cross_experiment_invariants": checks,
    "experiments": experiments,
    "global_thesis_rules": [
        "Dataset A and Dataset B must remain explicitly separate.",
        "Accuracy must not be used alone where macro metrics materially alter interpretation.",
        "Validation-only results must not be described as held-out test results.",
        "Offline mitigation-policy evaluation must not be described as live enforcement.",
        "CPU batch amortized inference timing must not be described as HTTP request latency.",
        "Feature-axis CNN-LSTM processing must not be described as genuine temporal sequence modeling.",
        "Controlled-laboratory perfect results must not be generalized to production or arbitrary DDoS attacks.",
        "Windows and requests nested within runs must not be treated as independent experimental replicates.",
        "Zero-support classes or attack families must be reported as not evaluated rather than zero performance.",
        "All final thesis numbers must trace to an artifact listed in this matrix."
    ]
}

OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "thesis_evidence_matrix.json"
path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")

print("EXPERIMENTS", len(experiments))
print("INVARIANTS", len(checks))
print("INVARIANTS_PASS", all(checks.values()))
print("OUTPUT", path)
print("SHA256", sha(path))

for x in experiments:
    print(x["experiment"], x["status"])
