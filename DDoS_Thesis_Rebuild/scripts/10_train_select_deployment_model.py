#!/usr/bin/env python3

import csv
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


SEED = 20261004

FROZEN_PROTOCOL_SHA256 = (
    "8d1508d284de5bd100bb01048970796ebe1c807d77b3fff2984927f9259b4736"
)

ROOT = Path(__file__).resolve().parents[1]

EXP_ROOT = (
    ROOT
    / "runs"
    / "experiment_010_realtime_detection_mitigation"
    / "deployment_model"
)

ACQ_ROOT = EXP_ROOT / "acquisition" / "runs"

PROTOCOL_PATH = (
    EXP_ROOT
    / "modeling_protocol_v1"
    / "modeling_protocol.json"
)

OUTPUT_ROOT = EXP_ROOT / "model_development_v1"

FORBIDDEN_TEST_RUNS = {"B07", "B08", "A07", "A08"}

FEATURE_FILE_NAMES = {
    10: [
        ("features_w10.csv", "features_w10_metadata.json"),
        ("windows_10.csv", "windows_10.metadata.json"),
        ("window_10.csv", "window_10_metadata.json"),
    ],
    100: [
        ("features_w100.csv", "features_w100_metadata.json"),
        ("windows_100.csv", "windows_100.metadata.json"),
        ("window_100.csv", "window_100_metadata.json"),
    ],
}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def fail(message):
    raise RuntimeError(message)


def set_determinism():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_protocol():
    actual = sha256_file(PROTOCOL_PATH)

    if actual != FROZEN_PROTOCOL_SHA256:
        fail(
            "Frozen modeling protocol hash mismatch: "
            f"{actual}"
        )

    with PROTOCOL_PATH.open(encoding="utf-8") as f:
        protocol = json.load(f)

    if protocol["protocol_version"] != "1.0":
        fail("Unexpected protocol version.")

    return protocol


def verify_protocol(protocol):
    pc = protocol["partition_contract"]

    expected_train = [
        "B01",
        "B02",
        "B03",
        "B04",
        "A01",
        "A02",
        "A03",
        "A04",
    ]

    expected_validation = [
        "B05",
        "B06",
        "A05",
        "A06",
    ]

    expected_test = [
        "B07",
        "B08",
        "A07",
        "A08",
    ]

    if pc["train_runs"] != expected_train:
        fail("Unexpected frozen training-run contract.")

    if pc["validation_runs"] != expected_validation:
        fail("Unexpected frozen validation-run contract.")

    if pc["test_runs"] != expected_test:
        fail("Unexpected frozen test-run contract.")

    if pc["partition_unit"] != "acquisition_run":
        fail("Partition unit is not acquisition_run.")

    if pc["random_window_split_permitted"]:
        fail("Protocol unexpectedly permits random window split.")

    if pc["cross_partition_run_permitted"]:
        fail("Protocol unexpectedly permits cross-partition runs.")

    if pc["pcap_reuse_across_partitions_permitted"]:
        fail("Protocol unexpectedly permits PCAP reuse.")

    if pc["test_used_for_model_selection"]:
        fail("Protocol unexpectedly permits test selection.")

    if pc["test_access_before_final_model_freeze"]:
        fail("Protocol unexpectedly permits pre-freeze test access.")

    if protocol["feature_contract"]["feature_count"] != 20:
        fail("Feature count is not 20.")

    if len(protocol["feature_contract"]["feature_order"]) != 20:
        fail("Feature-order length is not 20.")

    if protocol["assembly_contract"]["window_index_used_as_model_feature"]:
        fail("window_index unexpectedly permitted as feature.")

    if protocol["assembly_contract"]["metadata_used_as_model_features"]:
        fail("Metadata unexpectedly permitted as model features.")

    if protocol["assembly_contract"]["source_ip_used_as_model_feature"]:
        fail("source_ip unexpectedly permitted as model feature.")

    if protocol["assembly_contract"]["request_id_used_as_model_feature"]:
        fail("request_id unexpectedly permitted as model feature.")

    if protocol["assembly_contract"]["run_id_used_as_model_feature"]:
        fail("run_id unexpectedly permitted as model feature.")

    if protocol["training_contract"]["class_labels"] != {
        "Benign": 0,
        "Attack": 1,
    }:
        fail("Unexpected class-label contract.")

    if protocol["preprocessing"]["fit_scope"] != "training_partition_only":
        fail("Unexpected preprocessing fit scope.")

    if protocol["preprocessing"]["validation_refit_permitted"]:
        fail("Validation scaler refit unexpectedly permitted.")

    if protocol["preprocessing"]["test_refit_permitted"]:
        fail("Test scaler refit unexpectedly permitted.")

    if protocol["interpretation_constraints"]["hybrid_model_must_win"]:
        fail("Protocol unexpectedly requires hybrid to win.")


def discover_runs(protocol):
    wanted = (
        protocol["partition_contract"]["train_runs"]
        + protocol["partition_contract"]["validation_runs"]
    )

    found = {}

    if not ACQ_ROOT.exists():
        fail(f"Acquisition root does not exist: {ACQ_ROOT}")

    for directory in sorted(ACQ_ROOT.iterdir()):
        if not directory.is_dir():
            continue

        status_path = directory / "completion_status.json"

        if not status_path.exists():
            continue

        with status_path.open(encoding="utf-8") as f:
            status = json.load(f)

        rid = status.get("logical_run_id")

        if rid not in wanted:
            continue

        if rid in FORBIDDEN_TEST_RUNS:
            fail(f"Forbidden test run discovered as input: {rid}")

        if rid in found:
            fail(f"Duplicate logical run: {rid}")

        found[rid] = {
            "directory": directory,
            "status": status,
        }

    missing = sorted(set(wanted) - set(found))

    if missing:
        fail(f"Missing required development runs: {missing}")

    if set(found) != set(wanted):
        fail("Unexpected development run set.")

    return found


def locate_feature_pair(run_dir, window):
    hits = []

    for csv_name, metadata_name in FEATURE_FILE_NAMES[window]:
        csv_path = run_dir / "features" / csv_name
        metadata_path = run_dir / "features" / metadata_name

        if csv_path.exists() or metadata_path.exists():
            if not (csv_path.exists() and metadata_path.exists()):
                fail(
                    f"Unpaired feature artifact in "
                    f"{run_dir.name}, W{window}"
                )

            hits.append((csv_path, metadata_path))

    if len(hits) != 1:
        fail(
            f"Expected exactly one feature artifact pair in "
            f"{run_dir.name}, W{window}; found {len(hits)}"
        )

    return hits[0]


def validate_and_load_feature_csv(
    rid,
    run_info,
    window,
    features,
    expected_extractor_sha,
):
    run_dir = run_info["directory"]

    csv_path, metadata_path = locate_feature_pair(
        run_dir,
        window,
    )

    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("feature_order") != features:
        fail(f"{rid} W{window}: feature order mismatch")

    if metadata.get("feature_count") != len(features):
        fail(f"{rid} W{window}: feature count mismatch")

    if metadata.get("labels_supplied_to_extractor") is not False:
        fail(f"{rid} W{window}: extraction not label-blind")

    if metadata.get("window_size_packets") != window:
        fail(f"{rid} W{window}: window-size mismatch")

    if metadata.get("window_stride_packets") != window:
        fail(f"{rid} W{window}: stride mismatch")

    if metadata.get("window_overlap") is not False:
        fail(f"{rid} W{window}: overlap mismatch")

    if (
        metadata.get("extractor_script_sha256")
        != expected_extractor_sha
    ):
        fail(f"{rid} W{window}: extractor hash mismatch")

    actual_csv_sha = sha256_file(csv_path)

    if metadata.get("output_csv_sha256") != actual_csv_sha:
        fail(f"{rid} W{window}: CSV hash mismatch")

    df = pd.read_csv(
        csv_path,
        keep_default_na=True,
    )

    expected_columns = ["window_index"] + features

    if list(df.columns) != expected_columns:
        fail(f"{rid} W{window}: CSV schema mismatch")

    if df["window_index"].isna().any():
        fail(f"{rid} W{window}: null window_index")

    try:
        window_indices = pd.to_numeric(
            df["window_index"],
            errors="raise",
        ).to_numpy(dtype=np.int64)
    except Exception as exc:
        fail(
            f"{rid} W{window}: invalid window_index: {exc}"
        )

    expected_indices = np.arange(
        len(df),
        dtype=np.int64,
    )

    if not np.array_equal(
        window_indices,
        expected_indices,
    ):
        fail(f"{rid} W{window}: window_index mismatch")

    if len(df) != metadata.get("complete_window_count"):
        fail(f"{rid} W{window}: row-count mismatch")

    X = df[features].copy()

    for col in features:
        original = X[col]

        parsed = pd.to_numeric(
            original,
            errors="coerce",
        )

        malformed = original.notna() & parsed.isna()

        if malformed.any():
            fail(
                f"{rid} W{window}: "
                f"nonnumeric/corrupt value in {col}"
            )

        if col == "packet_rate":
            parsed = parsed.fillna(0.0)
        elif parsed.isna().any():
            fail(
                f"{rid} W{window}: invalid null in {col}"
            )

        arr = parsed.to_numpy(dtype=np.float64)

        if not np.isfinite(arr).all():
            fail(
                f"{rid} W{window}: "
                f"nonfinite value in {col}"
            )

        X[col] = arr

    return {
        "X": X.to_numpy(dtype=np.float64),
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "csv_sha256": actual_csv_sha,
        "metadata_sha256": sha256_file(metadata_path),
        "rows": len(df),
    }


def class_for_run(rid):
    if rid.startswith("B"):
        return "Benign", 0

    if rid.startswith("A"):
        return "Attack", 1

    fail(f"Unknown logical run ID: {rid}")


def verify_run_status(rid, run_info, partition):
    status = run_info["status"]

    klass, _ = class_for_run(rid)

    if status.get("class") != klass:
        fail(f"{rid}: status class mismatch")

    if status.get("partition") != partition:
        fail(f"{rid}: status partition mismatch")

    status_value = status.get("status")

    if status_value not in {
        "COMPLETE",
        "DATASET_B_RUN_COMPLETE",
    }:
        fail(
            f"{rid}: unexpected completion status "
            f"{status_value!r}"
        )


def preflight_inputs(protocol, runs):
    features = protocol["feature_contract"]["feature_order"]
    extractor_sha = protocol["feature_contract"][
        "frozen_extractor_sha256"
    ]

    train_runs = protocol["partition_contract"]["train_runs"]
    validation_runs = protocol["partition_contract"][
        "validation_runs"
    ]
    test_runs = set(
        protocol["partition_contract"]["test_runs"]
    )

    loaded_runs = set(runs)

    test_intersection = sorted(
        loaded_runs & test_runs
    )

    if test_intersection:
        fail(
            "Preflight detected forbidden test-run access: "
            f"{test_intersection}"
        )

    expected = set(
        train_runs + validation_runs
    )

    if loaded_runs != expected:
        fail(
            "Development-run set differs from frozen protocol."
        )

    report = {
        "loaded_logical_run_ids": sorted(loaded_runs),
        "test_logical_run_ids": sorted(test_runs),
        "test_run_intersection": test_intersection,
        "test_runs_accessed": bool(test_intersection),
        "windows": {},
    }

    for rid in train_runs:
        verify_run_status(
            rid,
            runs[rid],
            "train",
        )

    for rid in validation_runs:
        verify_run_status(
            rid,
            runs[rid],
            "validation",
        )

    for window in protocol["assembly_contract"][
        "candidate_window_sizes_packets"
    ]:
        report["windows"][str(window)] = {}

        for rid in train_runs + validation_runs:
            item = validate_and_load_feature_csv(
                rid,
                runs[rid],
                window,
                features,
                extractor_sha,
            )

            report["windows"][str(window)][rid] = {
                "rows": item["rows"],
                "feature_csv": str(
                    item["csv_path"].relative_to(ROOT)
                ),
                "feature_csv_sha256":
                    item["csv_sha256"],
                "feature_metadata": str(
                    item["metadata_path"].relative_to(ROOT)
                ),
                "feature_metadata_sha256":
                    item["metadata_sha256"],
            }

    return report


def assemble(protocol, runs, window):
    features = protocol["feature_contract"]["feature_order"]
    extractor_sha = protocol["feature_contract"][
        "frozen_extractor_sha256"
    ]

    train_runs = protocol["partition_contract"]["train_runs"]
    validation_runs = protocol["partition_contract"][
        "validation_runs"
    ]

    data = {}
    provenance = {}

    for rid in train_runs + validation_runs:
        partition = (
            "train"
            if rid in train_runs
            else "validation"
        )

        verify_run_status(
            rid,
            runs[rid],
            partition,
        )

        item = validate_and_load_feature_csv(
            rid,
            runs[rid],
            window,
            features,
            extractor_sha,
        )

        klass, label = class_for_run(rid)

        data[rid] = {
            "X": item["X"],
            "y": np.full(
                item["rows"],
                label,
                dtype=np.int64,
            ),
            "class": klass,
            "partition": partition,
        }

        provenance[rid] = {
            "class": klass,
            "partition": partition,
            "rows": item["rows"],
            "feature_csv": str(
                item["csv_path"].relative_to(ROOT)
            ),
            "feature_csv_sha256":
                item["csv_sha256"],
            "feature_metadata": str(
                item["metadata_path"].relative_to(ROOT)
            ),
            "feature_metadata_sha256":
                item["metadata_sha256"],
        }

    X_train = np.concatenate(
        [
            data[rid]["X"]
            for rid in train_runs
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            data[rid]["y"]
            for rid in train_runs
        ],
        axis=0,
    )

    return (
        data,
        X_train,
        y_train,
        provenance,
    )


def get_hybrid_architecture(protocol):
    spec = protocol["candidate_models"][
        "hybrid_cnn_lstm"
    ]

    architecture = spec.get("architecture")

    if not isinstance(architecture, list):
        fail("Hybrid architecture is not a list.")

    expected = [
        {
            "layer": "Conv1D",
            "filters": 32,
            "kernel_size": 3,
            "activation": "relu",
            "padding": "same",
        },
        {
            "layer": "MaxPooling1D",
            "pool_size": 2,
        },
        {
            "layer": "LSTM",
            "units": 32,
            "return_sequences": False,
        },
        {
            "layer": "Dense",
            "units": 16,
            "activation": "relu",
        },
        {
            "layer": "Dropout",
            "rate": 0.2,
        },
        {
            "layer": "Dense",
            "units": 1,
            "activation": "sigmoid",
        },
    ]

    if architecture != expected:
        fail(
            "Frozen hybrid architecture does not match "
            "the implementation contract."
        )

    return architecture


def build_hybrid(protocol):
    spec = protocol["candidate_models"][
        "hybrid_cnn_lstm"
    ]

    architecture = get_hybrid_architecture(
        protocol
    )

    input_shape = tuple(spec["input_shape"])

    feature_count = len(
        protocol["feature_contract"]["feature_order"]
    )

    if input_shape != (feature_count, 1):
        fail(
            "Hybrid input shape does not match "
            "the frozen feature contract."
        )

    conv = architecture[0]
    pool = architecture[1]
    lstm = architecture[2]
    dense = architecture[3]
    dropout = architecture[4]
    output = architecture[5]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=input_shape
            ),
            tf.keras.layers.Conv1D(
                filters=conv["filters"],
                kernel_size=conv["kernel_size"],
                activation=conv["activation"],
                padding=conv["padding"],
            ),
            tf.keras.layers.MaxPooling1D(
                pool_size=pool["pool_size"]
            ),
            tf.keras.layers.LSTM(
                units=lstm["units"],
                return_sequences=lstm[
                    "return_sequences"
                ],
            ),
            tf.keras.layers.Dense(
                units=dense["units"],
                activation=dense["activation"],
            ),
            tf.keras.layers.Dropout(
                rate=dropout["rate"]
            ),
            tf.keras.layers.Dense(
                units=output["units"],
                activation=output["activation"],
            ),
        ]
    )

    if spec["optimizer"] != "adam":
        fail("Unexpected hybrid optimizer.")

    if spec["loss"] != "binary_crossentropy":
        fail("Unexpected hybrid loss.")

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=spec["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss=spec["loss"],
    )

    return model


def per_run_metrics(
    y_true,
    probability,
    threshold,
):
    pred = (
        probability >= threshold
    ).astype(np.int64)

    if len(y_true) == 0:
        fail("Validation run contains zero rows.")

    present_class = int(y_true[0])

    if not np.all(y_true == present_class):
        fail(
            "Validation run unexpectedly contains "
            "multiple classes."
        )

    present_recall = recall_score(
        y_true,
        pred,
        pos_label=present_class,
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        pred,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )

    return {
        "balanced_accuracy":
            float(present_recall),
        "macro_f1":
            float(macro_f1),
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    pred,
                )
            ),
        "support":
            int(len(y_true)),
        "true_class":
            present_class,
        "predicted_attack_count":
            int(pred.sum()),
        "predicted_benign_count":
            int((pred == 0).sum()),
    }


def pooled_metrics(
    y_true,
    probability,
    threshold,
):
    pred = (
        probability >= threshold
    ).astype(np.int64)

    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    )

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),
        "precision_attack": float(
            precision_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_attack": float(
            recall_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_attack": float(
            f1_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "precision_benign": float(
            precision_score(
                y_true,
                pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "recall_benign": float(
            recall_score(
                y_true,
                pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_benign": float(
            f1_score(
                y_true,
                pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "confusion_matrix":
            cm.astype(int).tolist(),
        "support":
            int(len(y_true)),
    }


def evaluate_thresholds(
    model_name,
    window,
    probabilities,
    data,
    protocol,
):
    validation_runs = protocol[
        "partition_contract"
    ]["validation_runs"]

    thresholds = protocol[
        "threshold_policy"
    ]["probabilistic_models"][
        "candidate_thresholds"
    ]

    results = []

    pooled_y = np.concatenate(
        [
            data[rid]["y"]
            for rid in validation_runs
        ]
    )

    pooled_probability = np.concatenate(
        [
            probabilities[rid]
            for rid in validation_runs
        ]
    )

    for threshold in thresholds:
        run_metrics = {}

        for rid in validation_runs:
            run_metrics[rid] = per_run_metrics(
                data[rid]["y"],
                probabilities[rid],
                threshold,
            )

        mean_ba = float(
            np.mean(
                [
                    run_metrics[rid][
                        "balanced_accuracy"
                    ]
                    for rid in validation_runs
                ]
            )
        )

        mean_macro_f1 = float(
            np.mean(
                [
                    run_metrics[rid][
                        "macro_f1"
                    ]
                    for rid in validation_runs
                ]
            )
        )

        results.append(
            {
                "model": model_name,
                "window_size_packets":
                    window,
                "threshold":
                    float(threshold),
                "mean_per_run_balanced_accuracy":
                    mean_ba,
                "mean_per_run_macro_f1":
                    mean_macro_f1,
                "threshold_distance_from_0_5":
                    abs(
                        float(threshold)
                        - 0.5
                    ),
                "per_run_metrics":
                    run_metrics,
                "pooled_metrics":
                    pooled_metrics(
                        pooled_y,
                        pooled_probability,
                        threshold,
                    ),
            }
        )

    ranked = sorted(
        results,
        key=lambda x: (
            -x[
                "mean_per_run_balanced_accuracy"
            ],
            -x[
                "mean_per_run_macro_f1"
            ],
            x[
                "threshold_distance_from_0_5"
            ],
        ),
    )

    return results, ranked[0]


def save_predictions(
    path,
    probabilities,
    data,
    protocol,
):
    validation_runs = protocol[
        "partition_contract"
    ]["validation_runs"]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "logical_run_id",
                "class",
                "partition",
                "window_index",
                "y_true",
                "attack_probability",
            ]
        )

        for rid in validation_runs:
            probability = probabilities[rid]
            y_true = data[rid]["y"]

            if len(probability) != len(y_true):
                fail(
                    f"{rid}: prediction length mismatch"
                )

            if not np.isfinite(
                probability
            ).all():
                fail(
                    f"{rid}: nonfinite prediction probability"
                )

            if (
                (probability < 0.0).any()
                or (probability > 1.0).any()
            ):
                fail(
                    f"{rid}: probability outside [0,1]"
                )

            for index in range(len(y_true)):
                writer.writerow(
                    [
                        rid,
                        data[rid]["class"],
                        "validation",
                        index,
                        int(y_true[index]),
                        format(
                            float(
                                probability[index]
                            ),
                            ".17g",
                        ),
                    ]
                )


def train_candidate(
    model_name,
    window,
    protocol,
    data,
    X_train,
    y_train,
    candidate_dir,
):
    validation_runs = protocol[
        "partition_contract"
    ]["validation_runs"]

    candidate_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    scaler = None

    if model_name in {
        "logistic_regression",
        "hybrid_cnn_lstm",
    }:
        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(
            X_train
        )

        joblib.dump(
            scaler,
            candidate_dir / "scaler.joblib",
        )
    else:
        X_train_model = X_train

    if model_name == "logistic_regression":
        spec = protocol["candidate_models"][
            model_name
        ]

        model = LogisticRegression(
            C=spec["C"],
            penalty=spec["penalty"],
            solver=spec["solver"],
            class_weight=spec["class_weight"],
            max_iter=spec["max_iter"],
            random_state=spec["random_state"],
        )

        model.fit(
            X_train_model,
            y_train,
        )

        joblib.dump(
            model,
            candidate_dir / "model.joblib",
        )

        probabilities = {}

        for rid in validation_runs:
            X_validation = scaler.transform(
                data[rid]["X"]
            )

            probabilities[rid] = (
                model.predict_proba(
                    X_validation
                )[:, 1]
            )

    elif model_name == "random_forest":
        spec = protocol["candidate_models"][
            model_name
        ]

        model = RandomForestClassifier(
            n_estimators=
                spec["n_estimators"],
            max_depth=
                spec["max_depth"],
            min_samples_split=
                spec["min_samples_split"],
            min_samples_leaf=
                spec["min_samples_leaf"],
            max_features=
                spec["max_features"],
            class_weight=
                spec["class_weight"],
            bootstrap=
                spec["bootstrap"],
            n_jobs=
                spec["n_jobs"],
            random_state=
                spec["random_state"],
        )

        model.fit(
            X_train_model,
            y_train,
        )

        joblib.dump(
            model,
            candidate_dir / "model.joblib",
        )

        probabilities = {}

        for rid in validation_runs:
            probabilities[rid] = (
                model.predict_proba(
                    data[rid]["X"]
                )[:, 1]
            )

    elif model_name == "hybrid_cnn_lstm":
        spec = protocol["candidate_models"][
            model_name
        ]

        if (
            spec["input_semantics"]
            != "feature_axis_not_temporal_packet_sequence"
        ):
            fail(
                "Unexpected hybrid input semantics."
            )

        input_shape = tuple(
            spec["input_shape"]
        )

        feature_count = len(
            protocol["feature_contract"][
                "feature_order"
            ]
        )

        if input_shape != (
            feature_count,
            1,
        ):
            fail(
                "Hybrid input shape does not match "
                "the feature contract."
            )

        model = build_hybrid(
            protocol
        )

        X_train_hybrid = (
            X_train_model.reshape(
                (-1, *input_shape)
            )
        )

        X_validation = np.concatenate(
            [
                scaler.transform(
                    data[rid]["X"]
                )
                for rid in validation_runs
            ],
            axis=0,
        ).reshape(
            (-1, *input_shape)
        )

        y_validation = np.concatenate(
            [
                data[rid]["y"]
                for rid in validation_runs
            ],
            axis=0,
        )

        callback = (
            tf.keras.callbacks.EarlyStopping(
                monitor=
                    spec[
                        "early_stopping_monitor"
                    ],
                patience=
                    spec[
                        "early_stopping_patience"
                    ],
                restore_best_weights=
                    spec[
                        "restore_best_weights"
                    ],
            )
        )

        history = model.fit(
            X_train_hybrid,
            y_train,
            validation_data=(
                X_validation,
                y_validation,
            ),
            epochs=
                spec["maximum_epochs"],
            batch_size=
                spec["batch_size"],
            callbacks=[callback],
            verbose=2,
            shuffle=True,
        )

        history_obj = {
            key: [
                float(value)
                for value in values
            ]
            for key, values
            in history.history.items()
        }

        write_json(
            candidate_dir
            / "training_history.json",
            history_obj,
        )

        model.save(
            candidate_dir / "model.keras"
        )

        probabilities = {}

        for rid in validation_runs:
            X_run = scaler.transform(
                data[rid]["X"]
            ).reshape(
                (-1, *input_shape)
            )

            probabilities[rid] = (
                model.predict(
                    X_run,
                    batch_size=
                        spec["batch_size"],
                    verbose=0,
                )
                .reshape(-1)
                .astype(np.float64)
            )

    else:
        fail(
            f"Unknown model candidate: {model_name}"
        )

    prediction_path = (
        candidate_dir
        / "validation_predictions.csv"
    )

    save_predictions(
        prediction_path,
        probabilities,
        data,
        protocol,
    )

    all_thresholds, best = (
        evaluate_thresholds(
            model_name,
            window,
            probabilities,
            data,
            protocol,
        )
    )

    write_json(
        candidate_dir
        / "threshold_results.json",
        all_thresholds,
    )

    write_json(
        candidate_dir
        / "candidate_best.json",
        best,
    )

    return best


def artifact_manifest(directory):
    records = []

    for path in sorted(
        directory.rglob("*")
    ):
        if not path.is_file():
            continue

        if path.name in {
            "artifact_manifest.json",
            "artifact_manifest.sha256",
        }:
            continue

        records.append(
            {
                "path": str(
                    path.relative_to(
                        directory
                    )
                ),
                "bytes":
                    path.stat().st_size,
                "sha256":
                    sha256_file(path),
            }
        )

    return records


def main():
    if OUTPUT_ROOT.exists():
        fail(
            "Model-development output directory "
            "already exists. Refusing to overwrite "
            "scientific artifacts: "
            f"{OUTPUT_ROOT}"
        )

    set_determinism()

    protocol = load_protocol()

    verify_protocol(
        protocol
    )

    runs = discover_runs(
        protocol
    )

    print(
        "Running read-only "
        "development-input preflight..."
    )

    preflight = preflight_inputs(
        protocol,
        runs,
    )

    print(
        "Development-input preflight PASS"
    )

    if preflight["test_runs_accessed"]:
        fail(
            "Forbidden test-run access detected."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    executed_script = Path(
        __file__
    ).resolve()

    environment = {
        "python":
            sys.version.split()[0],
        "numpy":
            np.__version__,
        "pandas":
            pd.__version__,
        "sklearn":
            sklearn.__version__,
        "tensorflow":
            tf.__version__,
        "joblib":
            joblib.__version__,
        "tensorflow_gpu_devices": [
            device.name
            for device
            in tf.config.list_physical_devices(
                "GPU"
            )
        ],
        "random_seed":
            SEED,
        "protocol_sha256":
            sha256_file(
                PROTOCOL_PATH
            ),
        "executed_script_sha256":
            sha256_file(
                executed_script
            ),
    }

    write_json(
        OUTPUT_ROOT
        / "environment.json",
        environment,
    )

    write_json(
        OUTPUT_ROOT
        / "preflight_report.json",
        preflight,
    )

    shutil.copy2(
        PROTOCOL_PATH,
        OUTPUT_ROOT
        / "modeling_protocol.json",
    )

    shutil.copy2(
        executed_script,
        OUTPUT_ROOT
        / "10_train_select_deployment_model.py",
    )

    write_json(
        OUTPUT_ROOT
        / "implementation_provenance.json",
        {
            "executed_script":
                str(
                    executed_script.relative_to(
                        ROOT
                    )
                ),
            "executed_script_sha256":
                sha256_file(
                    executed_script
                ),
            "frozen_protocol":
                str(
                    PROTOCOL_PATH.relative_to(
                        ROOT
                    )
                ),
            "frozen_protocol_sha256":
                sha256_file(
                    PROTOCOL_PATH
                ),
        },
    )

    all_best = []
    input_provenance = {}

    models = [
        "logistic_regression",
        "random_forest",
        "hybrid_cnn_lstm",
    ]

    windows = protocol[
        "assembly_contract"
    ]["candidate_window_sizes_packets"]

    if windows != [10, 100]:
        fail(
            "Unexpected candidate window contract."
        )

    for window in windows:
        print(
            f"\n=== ASSEMBLING W{window} ==="
        )

        (
            data,
            X_train,
            y_train,
            provenance,
        ) = assemble(
            protocol,
            runs,
            window,
        )

        input_provenance[
            f"window_{window}"
        ] = provenance

        print(
            f"W{window} "
            f"training rows={len(y_train)} "
            f"benign={int((y_train == 0).sum())} "
            f"attack={int((y_train == 1).sum())}"
        )

        for model_name in models:
            print(
                f"\n=== TRAIN "
                f"{model_name} "
                f"W{window} ===",
                flush=True,
            )

            candidate_dir = (
                OUTPUT_ROOT
                / "candidates"
                / f"{model_name}_w{window}"
            )

            best = train_candidate(
                model_name,
                window,
                protocol,
                data,
                X_train,
                y_train,
                candidate_dir,
            )

            all_best.append(
                best
            )

            print(
                "candidate best:",
                model_name,
                f"W{window}",
                "threshold="
                + str(
                    best["threshold"]
                ),
                "mean_run_BA="
                + str(
                    best[
                        "mean_per_run_balanced_accuracy"
                    ]
                ),
                "mean_run_macroF1="
                + str(
                    best[
                        "mean_per_run_macro_f1"
                    ]
                ),
            )

            tf.keras.backend.clear_session()

    write_json(
        OUTPUT_ROOT
        / "input_provenance.json",
        input_provenance,
    )

    complexity = {
        "logistic_regression": 0,
        "random_forest": 1,
        "hybrid_cnn_lstm": 2,
    }

    ranked = sorted(
        all_best,
        key=lambda x: (
            -x[
                "mean_per_run_balanced_accuracy"
            ],
            -x[
                "mean_per_run_macro_f1"
            ],
            complexity[x["model"]],
            (
                0
                if x[
                    "window_size_packets"
                ] == 100
                else 1
            ),
        ),
    )

    if len(ranked) != 6:
        fail(
            "Expected exactly six candidate "
            "model/window systems."
        )

    selected = ranked[0]

    test_intersection = preflight[
        "test_run_intersection"
    ]

    if test_intersection:
        fail(
            "Cannot freeze selected system: "
            "test runs were accessed."
        )

    selection = {
        "status":
            "SELECTED_FROM_FROZEN_VALIDATION_PROTOCOL",
        "selected_model":
            selected["model"],
        "selected_window_size_packets":
            selected[
                "window_size_packets"
            ],
        "selected_threshold":
            selected["threshold"],
        "mean_per_run_balanced_accuracy":
            selected[
                "mean_per_run_balanced_accuracy"
            ],
        "mean_per_run_macro_f1":
            selected[
                "mean_per_run_macro_f1"
            ],
        "selection_ranking":
            ranked,
        "test_runs_accessed":
            bool(test_intersection),
        "loaded_development_runs":
            preflight[
                "loaded_logical_run_ids"
            ],
        "test_runs":
            protocol[
                "partition_contract"
            ]["test_runs"],
        "interpretation_scope":
            protocol[
                "interpretation_constraints"
            ]["claim_scope"],
    }

    write_json(
        OUTPUT_ROOT
        / "selected_system.json",
        selection,
    )

    manifest = artifact_manifest(
        OUTPUT_ROOT
    )

    write_json(
        OUTPUT_ROOT
        / "artifact_manifest.json",
        manifest,
    )

    manifest_hash = sha256_file(
        OUTPUT_ROOT
        / "artifact_manifest.json"
    )

    (
        OUTPUT_ROOT
        / "artifact_manifest.sha256"
    ).write_text(
        manifest_hash
        + "  artifact_manifest.json\n",
        encoding="utf-8",
    )

    print(
        "\n========================================"
    )
    print(
        "MODEL DEVELOPMENT COMPLETE"
    )
    print(
        "========================================"
    )
    print(
        "selected_model =",
        selected["model"],
    )
    print(
        "selected_window_size_packets =",
        selected[
            "window_size_packets"
        ],
    )
    print(
        "selected_threshold =",
        selected["threshold"],
    )
    print(
        "mean_per_run_balanced_accuracy =",
        selected[
            "mean_per_run_balanced_accuracy"
        ],
    )
    print(
        "mean_per_run_macro_f1 =",
        selected[
            "mean_per_run_macro_f1"
        ],
    )
    print(
        "test_runs_accessed =",
        bool(test_intersection),
    )
    print(
        "output =",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()