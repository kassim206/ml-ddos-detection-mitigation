#!/usr/bin/env python3
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/experiment_010_realtime_detection_mitigation/deployment_model"
FREEZE = BASE / "selected_system_freeze"
ACQ = BASE / "acquisition/runs"
OUT = BASE / "final_test_evaluation_v1"

FEATURES = [
    "packet_count",
    "duration_seconds",
    "header_length_mean",
    "ttl_mean",
    "fin_fraction",
    "syn_fraction",
    "rst_fraction",
    "psh_fraction",
    "ack_fraction",
    "packet_length_sum",
    "packet_length_min",
    "packet_length_max",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "iat_mean",
    "iat_min",
    "iat_max",
    "iat_std",
    "packet_rate",
]

RUNS = {
    "B07": {"label": 0, "class": "Benign"},
    "B08": {"label": 0, "class": "Benign"},
    "A07": {"label": 1, "class": "Attack"},
    "A08": {"label": 1, "class": "Attack"},
}

EXPECTED = {
    "selected_system.json": "907f2288a579ed849b9186bb9258b692868621ff1bb62c17a3c45ab367020eee",
    "model.joblib": "5d07101d649141869c6da038b5f5c6d0f64a9dcf16b19c0d85ddc32eecb56fc9",
    "scaler.joblib": "3ae7e8c2e8c669bc5535eec7a9e902ff0ee821fb3debab6189a847e0e7da8fa0",
    "modeling_protocol.json": "8d1508d284de5bd100bb01048970796ebe1c807d77b3fff2984927f9259b4736",
}

THRESHOLD = 0.5
WINDOW_SIZE = 100


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_run(logical_id):
    matches = sorted(ACQ.glob(f"exp010_dataset_b_{logical_id}_*"))
    if len(matches) != 1:
        raise RuntimeError(f"{logical_id}: expected exactly one acquisition directory, found {len(matches)}")
    return matches[0]


def binary_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    accuracy = float(np.mean(y_true == y_pred))
    recall_benign = float(r[0])
    recall_attack = float(r[1])
    balanced = float((recall_benign + recall_attack) / 2.0)
    return {
        "support": int(len(y_true)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision_benign": float(p[0]),
        "recall_benign": recall_benign,
        "f1_benign": float(f[0]),
        "precision_attack": float(p[1]),
        "recall_attack": recall_attack,
        "f1_attack": float(f[1]),
        "macro_f1": float(np.mean(f)),
        "confusion_matrix": cm.tolist(),
    }


if OUT.exists():
    raise RuntimeError(f"refusing to overwrite existing output directory: {OUT}")

for name, expected_hash in EXPECTED.items():
    path = FREEZE / name
    if not path.is_file():
        raise RuntimeError(f"missing frozen selected-system artifact: {path}")
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"frozen artifact hash mismatch: {name}")

selected = json.loads((FREEZE / "selected_system.json").read_text(encoding="utf-8"))

if selected.get("status") != "SELECTED_FROM_FROZEN_VALIDATION_PROTOCOL":
    raise RuntimeError("unexpected selected-system status")
if selected.get("selected_model") != "logistic_regression":
    raise RuntimeError("selected model is not logistic_regression")
if selected.get("selected_window_size_packets") != WINDOW_SIZE:
    raise RuntimeError("selected window is not W100")
if float(selected.get("selected_threshold")) != THRESHOLD:
    raise RuntimeError("selected threshold is not 0.50")
if selected.get("test_runs_accessed") is not False:
    raise RuntimeError("selected-system record does not preserve untouched-test state")
if set(selected.get("test_runs", [])) != set(RUNS):
    raise RuntimeError("selected-system test-run set mismatch")

resolved = {}

for logical_id, spec in RUNS.items():
    run_dir = find_run(logical_id)
    completion_path = run_dir / "completion_status.json"
    feature_path = run_dir / "features/window_100.csv"
    metadata_path = run_dir / "features/window_100_metadata.json"

    if not completion_path.is_file():
        raise RuntimeError(f"{logical_id}: missing completion_status.json")
    if not feature_path.is_file():
        raise RuntimeError(f"{logical_id}: missing W100 feature CSV")
    if not metadata_path.is_file():
        raise RuntimeError(f"{logical_id}: missing W100 metadata")

    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if completion.get("status") != "COMPLETE":
        raise RuntimeError(f"{logical_id}: run is not COMPLETE")
    if completion.get("logical_run_id") != logical_id:
        raise RuntimeError(f"{logical_id}: completion logical ID mismatch")
    if completion.get("partition") != "test":
        raise RuntimeError(f"{logical_id}: run is not test partition")
    if completion.get("class") != spec["class"]:
        raise RuntimeError(f"{logical_id}: class mismatch")
    if completion.get("test_use_permitted") is not True:
        raise RuntimeError(f"{logical_id}: test use not permitted")
    if completion.get("training_use_permitted") is not False:
        raise RuntimeError(f"{logical_id}: training use unexpectedly permitted")
    if completion.get("validation_use_permitted") is not False:
        raise RuntimeError(f"{logical_id}: validation use unexpectedly permitted")
    if completion.get("raw_evidence_frozen") is not True:
        raise RuntimeError(f"{logical_id}: raw evidence not frozen")
    if completion.get("feature_artifacts_frozen") is not True:
        raise RuntimeError(f"{logical_id}: feature artifacts not frozen")

    if metadata.get("window_size_packets") != WINDOW_SIZE:
        raise RuntimeError(f"{logical_id}: wrong feature window")
    if metadata.get("window_stride_packets") != WINDOW_SIZE:
        raise RuntimeError(f"{logical_id}: wrong feature stride")
    if metadata.get("window_overlap") is not False:
        raise RuntimeError(f"{logical_id}: overlapping windows")
    if metadata.get("cross_capture_windows") is not False:
        raise RuntimeError(f"{logical_id}: cross-capture windows detected")
    if metadata.get("feature_count") != 20:
        raise RuntimeError(f"{logical_id}: feature count mismatch")
    if metadata.get("feature_order") != FEATURES:
        raise RuntimeError(f"{logical_id}: feature order mismatch")
    if metadata.get("labels_supplied_to_extractor") is not False:
        raise RuntimeError(f"{logical_id}: labels were supplied to extractor")
    if metadata.get("output_csv_sha256") != sha256(feature_path):
        raise RuntimeError(f"{logical_id}: W100 CSV hash mismatch")

    df = pd.read_csv(feature_path)

    if list(df.columns) != ["window_index"] + FEATURES:
        raise RuntimeError(f"{logical_id}: W100 CSV schema mismatch")
    if len(df) != 120:
        raise RuntimeError(f"{logical_id}: expected 120 W100 windows, found {len(df)}")
    if df["window_index"].tolist() != list(range(120)):
        raise RuntimeError(f"{logical_id}: non-sequential window indices")

    X = df[FEATURES].copy()

    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    non_packet_rate = [c for c in FEATURES if c != "packet_rate"]

    if X[non_packet_rate].isna().any().any():
        raise RuntimeError(f"{logical_id}: invalid null/non-numeric value outside packet_rate")

    values = X[non_packet_rate].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{logical_id}: non-finite value outside packet_rate")

    packet_rate = X["packet_rate"].to_numpy(dtype=np.float64)
    packet_rate = np.where(np.isnan(packet_rate), 0.0, packet_rate)

    if not np.isfinite(packet_rate).all():
        raise RuntimeError(f"{logical_id}: non-finite packet_rate after permitted null mapping")

    X["packet_rate"] = packet_rate

    resolved[logical_id] = {
        "run_dir": run_dir,
        "feature_path": feature_path,
        "metadata_path": metadata_path,
        "completion_path": completion_path,
        "X": X,
        "label": spec["label"],
        "class": spec["class"],
    }

scaler = joblib.load(FREEZE / "scaler.joblib")
model = joblib.load(FREEZE / "model.joblib")

if not hasattr(model, "predict_proba"):
    raise RuntimeError("selected model does not expose predict_proba")

OUT.mkdir(parents=False, exist_ok=False)

shutil.copy2(Path(__file__), OUT / Path(__file__).name)
shutil.copy2(FREEZE / "selected_system.json", OUT / "selected_system.json")
shutil.copy2(FREEZE / "modeling_protocol.json", OUT / "modeling_protocol.json")

all_true = []
all_pred = []
all_prob = []
per_run = {}
input_provenance = {}

for logical_id in ["B07", "B08", "A07", "A08"]:
    item = resolved[logical_id]
    X = item["X"].to_numpy(dtype=np.float64)
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)[:, 1]
    pred = (prob >= THRESHOLD).astype(np.int64)
    true = np.full(len(pred), item["label"], dtype=np.int64)

    if not np.isfinite(prob).all():
        raise RuntimeError(f"{logical_id}: non-finite model probabilities")

    present_recall = float(np.mean(pred == true))

    _, _, f, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )

    metrics = {
        "support": int(len(true)),
        "true_class": int(item["label"]),
        "accuracy": present_recall,
        "balanced_accuracy": present_recall,
        "macro_f1": float(np.mean(f)),
        "predicted_benign_count": int(np.sum(pred == 0)),
        "predicted_attack_count": int(np.sum(pred == 1)),
    }

    per_run[logical_id] = metrics

    pred_path = OUT / f"{logical_id}_predictions.csv"

    with pred_path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(
            [
                "logical_run_id",
                "window_index",
                "true_label",
                "attack_probability",
                "predicted_label",
            ]
        )
        for idx, (yt, pr, yp) in enumerate(zip(true, prob, pred)):
            writer.writerow(
                [
                    logical_id,
                    idx,
                    int(yt),
                    format(float(pr), ".17g"),
                    int(yp),
                ]
            )

    input_provenance[logical_id] = {
        "physical_run_id": item["run_dir"].name,
        "class": item["class"],
        "true_label": int(item["label"]),
        "window_size_packets": WINDOW_SIZE,
        "feature_csv": str(item["feature_path"].relative_to(ROOT)),
        "feature_csv_sha256": sha256(item["feature_path"]),
        "feature_metadata_sha256": sha256(item["metadata_path"]),
        "completion_status_sha256": sha256(item["completion_path"]),
        "prediction_csv": pred_path.name,
        "prediction_csv_sha256": sha256(pred_path),
    }

    all_true.append(true)
    all_pred.append(pred)
    all_prob.append(prob)

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)
y_prob = np.concatenate(all_prob)

pooled = binary_metrics(y_true, y_pred)

protocol_summary = {
    "mean_per_run_balanced_accuracy": float(
        np.mean([per_run[r]["balanced_accuracy"] for r in per_run])
    ),
    "mean_per_run_macro_f1": float(
        np.mean([per_run[r]["macro_f1"] for r in per_run])
    ),
}

with (OUT / "pooled_predictions.csv").open("w", newline="", encoding="utf-8") as fobj:
    writer = csv.writer(fobj)
    writer.writerow(
        [
            "logical_run_id",
            "window_index",
            "true_label",
            "attack_probability",
            "predicted_label",
        ]
    )
    for logical_id in ["B07", "B08", "A07", "A08"]:
        pred_path = OUT / f"{logical_id}_predictions.csv"
        with pred_path.open(newline="", encoding="utf-8") as src:
            reader = csv.DictReader(src)
            for row in reader:
                writer.writerow(
                    [
                        row["logical_run_id"],
                        row["window_index"],
                        row["true_label"],
                        row["attack_probability"],
                        row["predicted_label"],
                    ]
                )

results = {
    "status": "FINAL_TEST_EVALUATED_ONCE",
    "experiment": "Exp010",
    "dataset": "Dataset B",
    "interpretation_scope": selected["interpretation_scope"],
    "selected_model": "logistic_regression",
    "window_size_packets": WINDOW_SIZE,
    "threshold": THRESHOLD,
    "test_runs": ["B07", "B08", "A07", "A08"],
    "total_test_windows": int(len(y_true)),
    "per_run_metrics": per_run,
    "protocol_summary": protocol_summary,
    "pooled_metrics": pooled,
}

environment = {
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "joblib": joblib.__version__,
}

implementation_provenance = {
    "evaluation_script": str(Path(__file__).relative_to(ROOT)),
    "evaluation_script_sha256": sha256(Path(__file__)),
    "selected_system_sha256": EXPECTED["selected_system.json"],
    "model_sha256": EXPECTED["model.joblib"],
    "scaler_sha256": EXPECTED["scaler.joblib"],
    "modeling_protocol_sha256": EXPECTED["modeling_protocol.json"],
    "model_retrained_during_final_test": False,
    "scaler_refit_during_final_test": False,
    "threshold_changed_during_final_test": False,
    "window_changed_during_final_test": False,
    "model_reselected_during_final_test": False,
}

write_json(OUT / "final_test_results.json", results)
write_json(OUT / "input_provenance.json", input_provenance)
write_json(OUT / "environment.json", environment)
write_json(OUT / "implementation_provenance.json", implementation_provenance)

artifact_paths = sorted(
    p for p in OUT.iterdir()
    if p.is_file() and p.name not in {"artifact_manifest.json", "artifact_manifest.sha256"}
)

artifact_manifest = {
    str(p.relative_to(OUT)): sha256(p)
    for p in artifact_paths
}

write_json(OUT / "artifact_manifest.json", artifact_manifest)

(OUT / "artifact_manifest.sha256").write_text(
    f"{sha256(OUT / 'artifact_manifest.json')}  artifact_manifest.json\n",
    encoding="utf-8",
)

print("FINAL TEST EVALUATION COMPLETE")
print(json.dumps(results, indent=2, sort_keys=True))
print(f"artifact_manifest_sha256={sha256(OUT / 'artifact_manifest.json')}")
