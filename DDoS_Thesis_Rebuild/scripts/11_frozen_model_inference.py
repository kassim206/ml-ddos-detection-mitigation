#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np


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

MODEL_FREEZE = Path(
    "runs/experiment_010_realtime_detection_mitigation/"
    "deployment_model/selected_system_freeze"
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

THRESHOLD = 0.5
WINDOW_SIZE = 100


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_system():
    selected_path = MODEL_FREEZE / "selected_system.json"
    model_path = MODEL_FREEZE / "model.joblib"
    scaler_path = MODEL_FREEZE / "scaler.joblib"

    if sha256_file(selected_path) != EXPECTED_SELECTED_SYSTEM_SHA256:
        raise RuntimeError("Frozen selected_system.json hash mismatch")

    if sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Frozen model.joblib hash mismatch")

    if sha256_file(scaler_path) != EXPECTED_SCALER_SHA256:
        raise RuntimeError("Frozen scaler.joblib hash mismatch")

    selected = json.loads(selected_path.read_text())

    if selected["selected_model"] != "logistic_regression":
        raise RuntimeError("Unexpected selected model")

    if int(selected["selected_window_size_packets"]) != WINDOW_SIZE:
        raise RuntimeError("Unexpected selected window size")

    if float(selected["selected_threshold"]) != THRESHOLD:
        raise RuntimeError("Unexpected selected threshold")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if type(model).__name__ != "LogisticRegression":
        raise RuntimeError("Unexpected frozen model type")

    if type(scaler).__name__ != "StandardScaler":
        raise RuntimeError("Unexpected frozen scaler type")

    if list(model.classes_) != [0, 1]:
        raise RuntimeError("Frozen model class order is not [0, 1]")

    if int(model.n_features_in_) != len(FEATURES):
        raise RuntimeError("Frozen model feature count mismatch")

    if int(scaler.n_features_in_) != len(FEATURES):
        raise RuntimeError("Frozen scaler feature count mismatch")

    return model, scaler


def load_features(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)

        expected = ["window_index"] + FEATURES

        if reader.fieldnames != expected:
            raise RuntimeError(
                f"Feature schema mismatch: {reader.fieldnames}"
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError("Feature CSV contains no complete windows")

    matrix = []
    indices = []

    for expected_index, row in enumerate(rows):
        index = int(row["window_index"])

        if index != expected_index:
            raise RuntimeError("Window indices are not contiguous from zero")

        values = []

        for feature in FEATURES:
            value = float(row[feature])

            if not math.isfinite(value):
                raise RuntimeError(
                    f"Nonfinite feature at window {index}: {feature}"
                )

            values.append(value)

        indices.append(index)
        matrix.append(values)

    return indices, np.asarray(matrix, dtype=np.float64)


def infer(feature_csv):
    model, scaler = verify_frozen_system()
    indices, x = load_features(feature_csv)

    transformed = scaler.transform(x)

    if not np.isfinite(transformed).all():
        raise RuntimeError("Nonfinite value after frozen scaling")

    probabilities = model.predict_proba(transformed)[:, 1]

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Nonfinite attack probability")

    predictions = (probabilities >= THRESHOLD).astype(int)

    return indices, probabilities, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    indices, probabilities, predictions = infer(args.features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_index",
                "attack_probability",
                "predicted_label",
            ]
        )

        for index, probability, prediction in zip(
            indices,
            probabilities,
            predictions,
        ):
            writer.writerow(
                [
                    index,
                    format(float(probability), ".17g"),
                    int(prediction),
                ]
            )

    metadata = {
        "experiment": "Exp011",
        "purpose": "pre_result_frozen_model_inference_validation",
        "feature_csv": str(args.features),
        "feature_csv_sha256": sha256_file(args.features),
        "output_csv": str(args.output),
        "output_csv_sha256": sha256_file(args.output),
        "window_count": len(indices),
        "feature_count": len(FEATURES),
        "feature_order": FEATURES,
        "window_size_packets": WINDOW_SIZE,
        "selected_model": "logistic_regression",
        "selected_threshold": THRESHOLD,
        "positive_class": 1,
        "selected_system_sha256": EXPECTED_SELECTED_SYSTEM_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "implementation": str(Path(__file__)),
        "implementation_sha256": sha256_file(Path(__file__)),
        "labels_supplied_to_inference": False,
    }

    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
