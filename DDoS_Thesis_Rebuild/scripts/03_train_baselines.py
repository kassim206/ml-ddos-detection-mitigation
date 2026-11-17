#!/usr/bin/env python3

import gc
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "runs" / "experiment_002_preprocessing"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "experiment_003_baselines"
MEMMAP_DIR = OUTPUT_DIR / "training_memmap"

SCHEMA_FILE = (
    PROJECT_ROOT
    / "runs"
    / "experiment_001_dataset_char"
    / "schema_discovery_report.json"
)

TRAIN_FILE = DATA_DIR / "train_processed.parquet"
VAL_FILE = DATA_DIR / "val_processed.parquet"
TEST_FILE = DATA_DIR / "test_processed.parquet"

EXPECTED_TRAIN_ROWS = 7_702_608
EXPECTED_VAL_ROWS = 1_236_682
EXPECTED_TEST_ROWS = 1_773_902
EXPECTED_TOTAL_ROWS = 10_713_192
EXPECTED_FEATURE_COUNT = 39

RANDOM_SEED = 42

PARQUET_BATCH_SIZE = 25_000
PREDICTION_BATCH_SIZE = 25_000

RF_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 6

MODEL_N_JOBS = 1


def fail(message):
    print()
    print("=" * 70)
    print("[FAIL] Experiment 003 stopped")
    print("=" * 70)
    print(message)
    print("=" * 70)
    sys.exit(1)


def verify_file(path, description):
    if not path.exists():
        fail(f"Missing {description}: {path}")

    if path.stat().st_size == 0:
        fail(f"Empty {description}: {path}")


def release_memory(*objects):
    del objects
    gc.collect()


def get_parquet_metadata(path):
    try:
        return pq.ParquetFile(path)
    except Exception as exc:
        fail(
            f"Unable to open Parquet file:\n"
            f"{path}\n"
            f"Error: {exc}"
        )


def verify_parquet_structure(
    path,
    expected_rows,
    split_name,
    feature_columns,
):
    parquet_file = get_parquet_metadata(path)

    actual_rows = int(parquet_file.metadata.num_rows)

    print(
        f"    {split_name}: {actual_rows:,} rows"
    )

    if actual_rows != expected_rows:
        fail(
            f"{split_name} row-count mismatch.\n"
            f"Expected: {expected_rows:,}\n"
            f"Actual:   {actual_rows:,}"
        )

    available_columns = parquet_file.schema_arrow.names

    required_columns = list(feature_columns) + ["Category"]

    missing = [
        column
        for column in required_columns
        if column not in available_columns
    ]

    if missing:
        fail(
            f"{split_name} is missing required columns:\n"
            f"{missing}"
        )


def read_label_counts(path, split_name):
    parquet_file = get_parquet_metadata(path)

    counts = {}

    for arrow_batch in parquet_file.iter_batches(
        batch_size=PARQUET_BATCH_SIZE,
        columns=["Category"],
    ):
        batch_df = arrow_batch.to_pandas()

        if batch_df["Category"].isnull().any():
            fail(
                f"Missing Category labels detected in "
                f"{split_name}."
            )

        values = (
            batch_df["Category"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        for category, count in values.items():
            counts[category] = (
                counts.get(category, 0)
                + int(count)
            )

        release_memory(
            arrow_batch,
            batch_df,
        )

    return dict(sorted(counts.items()))


def validate_category_coverage():
    train_counts = read_label_counts(
        TRAIN_FILE,
        "train",
    )

    val_counts = read_label_counts(
        VAL_FILE,
        "val",
    )

    test_counts = read_label_counts(
        TEST_FILE,
        "test",
    )

    train_categories = set(train_counts)
    val_categories = set(val_counts)
    test_categories = set(test_counts)

    if train_categories != val_categories:
        fail(
            "Train and validation category sets differ.\n"
            f"Train: {sorted(train_categories)}\n"
            f"Val:   {sorted(val_categories)}"
        )

    if train_categories != test_categories:
        fail(
            "Train and test category sets differ.\n"
            f"Train: {sorted(train_categories)}\n"
            f"Test:  {sorted(test_categories)}"
        )

    categories = sorted(train_categories)

    if not categories:
        fail("No categories were found.")

    print()
    print("    Categories found in actual dataset:")

    for category in categories:
        print(
            f"      {category}: "
            f"train={train_counts.get(category, 0):,}, "
            f"val={val_counts.get(category, 0):,}, "
            f"test={test_counts.get(category, 0):,}"
        )

    return categories


def load_schema():
    verify_file(
        SCHEMA_FILE,
        "schema discovery report",
    )

    try:
        with open(
            SCHEMA_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            schema_report = json.load(f)
    except Exception as exc:
        fail(
            f"Unable to read schema discovery report.\n"
            f"Error: {exc}"
        )

    if "features" not in schema_report:
        fail(
            "schema_discovery_report.json does not contain "
            "a 'features' field."
        )

    feature_columns = schema_report["features"]

    if not isinstance(feature_columns, list):
        fail(
            "The 'features' field is not a list."
        )

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        fail(
            f"Expected exactly {EXPECTED_FEATURE_COUNT} "
            f"features but found {len(feature_columns)}."
        )

    if len(set(feature_columns)) != len(feature_columns):
        fail(
            "Duplicate feature names detected."
        )

    if "Category" in feature_columns:
        fail(
            "'Category' must not appear in the feature list."
        )

    return feature_columns


def create_training_memmaps():
    MEMMAP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_path = MEMMAP_DIR / "X_train.dat"
    y_path = MEMMAP_DIR / "y_train.dat"

    expected_x_bytes = (
        EXPECTED_TRAIN_ROWS
        * EXPECTED_FEATURE_COUNT
        * np.dtype(np.float32).itemsize
    )

    expected_y_bytes = (
        EXPECTED_TRAIN_ROWS
        * np.dtype(np.int32).itemsize
    )

    recreate_x = (
        not x_path.exists()
        or x_path.stat().st_size != expected_x_bytes
    )

    recreate_y = (
        not y_path.exists()
        or y_path.stat().st_size != expected_y_bytes
    )

    if recreate_x:
        if x_path.exists():
            x_path.unlink()

        X_train = np.memmap(
            x_path,
            dtype=np.float32,
            mode="w+",
            shape=(
                EXPECTED_TRAIN_ROWS,
                EXPECTED_FEATURE_COUNT,
            ),
        )
    else:
        X_train = np.memmap(
            x_path,
            dtype=np.float32,
            mode="r+",
            shape=(
                EXPECTED_TRAIN_ROWS,
                EXPECTED_FEATURE_COUNT,
            ),
        )

    if recreate_y:
        if y_path.exists():
            y_path.unlink()

        y_train = np.memmap(
            y_path,
            dtype=np.int32,
            mode="w+",
            shape=(EXPECTED_TRAIN_ROWS,),
        )
    else:
        y_train = np.memmap(
            y_path,
            dtype=np.int32,
            mode="r+",
            shape=(EXPECTED_TRAIN_ROWS,),
        )

    return X_train, y_train, recreate_x or recreate_y


def build_training_memmaps(
    feature_columns,
    label_encoder,
):
    print()
    print(
        f"    Preparing disk-backed training arrays "
        f"for {EXPECTED_TRAIN_ROWS:,} rows..."
    )

    X_train, y_train, needs_population = (
        create_training_memmaps()
    )

    if not needs_population:
        print(
            "    Existing valid training memmaps found."
        )
        print(
            "    Training data will be read from disk-backed arrays."
        )
        return X_train, y_train

    print(
        "    Reading Parquet training data in batches..."
    )

    parquet_file = get_parquet_metadata(
        TRAIN_FILE
    )

    columns = list(feature_columns) + ["Category"]

    row_position = 0
    batch_number = 0

    start = time.perf_counter()

    for arrow_batch in parquet_file.iter_batches(
        batch_size=PARQUET_BATCH_SIZE,
        columns=columns,
    ):
        batch_number += 1

        batch_df = arrow_batch.to_pandas()

        if batch_df.empty:
            release_memory(
                arrow_batch,
                batch_df,
            )
            continue

        batch_rows = len(batch_df)

        end_position = (
            row_position + batch_rows
        )

        if end_position > EXPECTED_TRAIN_ROWS:
            fail(
                "Training data exceeded expected row count."
            )

        if batch_df[feature_columns].isnull().any().any():
            fail(
                f"Missing feature values detected "
                f"in training batch {batch_number}."
            )

        if batch_df["Category"].isnull().any():
            fail(
                f"Missing labels detected "
                f"in training batch {batch_number}."
            )

        X_batch = batch_df[
            feature_columns
        ].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        y_batch = label_encoder.transform(
            batch_df["Category"].astype(str)
        ).astype(
            np.int32,
            copy=False,
        )

        if X_batch.shape != (
            batch_rows,
            EXPECTED_FEATURE_COUNT,
        ):
            fail(
                f"Unexpected feature shape in training "
                f"batch {batch_number}.\n"
                f"Expected rows: {batch_rows:,}\n"
                f"Actual shape: {X_batch.shape}"
            )

        if not np.isfinite(X_batch).all():
            fail(
                f"Non-finite values detected in "
                f"training batch {batch_number}."
            )

        X_train[
            row_position:end_position
        ] = X_batch

        y_train[
            row_position:end_position
        ] = y_batch

        row_position = end_position

        if (
            batch_number == 1
            or batch_number % 20 == 0
        ):
            print(
                f"      Loaded: "
                f"{row_position:,}/"
                f"{EXPECTED_TRAIN_ROWS:,}"
            )

        release_memory(
            arrow_batch,
            batch_df,
            X_batch,
            y_batch,
        )

    X_train.flush()
    y_train.flush()

    elapsed = time.perf_counter() - start

    if row_position != EXPECTED_TRAIN_ROWS:
        fail(
            "Training memmap population row count mismatch.\n"
            f"Expected: {EXPECTED_TRAIN_ROWS:,}\n"
            f"Actual:   {row_position:,}"
        )

    print(
        f"    Training data loaded in "
        f"{elapsed:.2f} seconds."
    )

    print(
        "    Disk-backed training arrays verified."
    )

    return X_train, y_train


def evaluate_model(
    model,
    model_name,
    feature_columns,
    label_encoder,
):
    print()
    print(
        f"    Evaluating {model_name} "
        f"on {EXPECTED_TEST_ROWS:,} test rows..."
    )

    parquet_file = get_parquet_metadata(
        TEST_FILE
    )

    number_of_classes = len(
        label_encoder.classes_
    )

    cm = np.zeros(
        (
            number_of_classes,
            number_of_classes,
        ),
        dtype=np.int64,
    )

    total_predictions = 0
    total_correct = 0

    start = time.perf_counter()

    columns = list(feature_columns) + ["Category"]

    batch_number = 0

    for arrow_batch in parquet_file.iter_batches(
        batch_size=PREDICTION_BATCH_SIZE,
        columns=columns,
    ):
        batch_number += 1

        batch_df = arrow_batch.to_pandas()

        if batch_df.empty:
            release_memory(
                arrow_batch,
                batch_df,
            )
            continue

        if batch_df[feature_columns].isnull().any().any():
            fail(
                f"Missing feature values detected "
                f"in test batch {batch_number}."
            )

        if batch_df["Category"].isnull().any():
            fail(
                f"Missing Category labels detected "
                f"in test batch {batch_number}."
            )

        X_batch = batch_df[
            feature_columns
        ].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        y_batch = label_encoder.transform(
            batch_df["Category"].astype(str)
        ).astype(
            np.int32,
            copy=False,
        )

        if not np.isfinite(X_batch).all():
            fail(
                f"Non-finite feature values detected "
                f"in test batch {batch_number}."
            )

        predictions = np.asarray(
            model.predict(X_batch),
            dtype=np.int32,
        )

        if len(predictions) != len(y_batch):
            fail(
                f"{model_name} prediction count mismatch "
                f"in batch {batch_number}."
            )

        cm += confusion_matrix(
            y_batch,
            predictions,
            labels=np.arange(number_of_classes),
        )

        total_predictions += len(predictions)

        total_correct += int(
            np.sum(predictions == y_batch)
        )

        release_memory(
            arrow_batch,
            batch_df,
            X_batch,
            y_batch,
            predictions,
        )

        if (
            batch_number == 1
            or batch_number % 20 == 0
        ):
            print(
                f"      Processed: "
                f"{total_predictions:,}/"
                f"{EXPECTED_TEST_ROWS:,}"
            )

    prediction_seconds = (
        time.perf_counter() - start
    )

    if total_predictions != EXPECTED_TEST_ROWS:
        fail(
            f"{model_name} prediction count does not "
            f"equal expected test rows.\n"
            f"Expected: {EXPECTED_TEST_ROWS:,}\n"
            f"Actual:   {total_predictions:,}"
        )

    if cm.sum() != EXPECTED_TEST_ROWS:
        fail(
            f"{model_name} confusion matrix total does "
            f"not equal expected test rows."
        )

    accuracy = (
        total_correct / total_predictions
    )

    precision_values = []
    recall_values = []
    f1_values = []

    for index in range(number_of_classes):
        tp = cm[index, index]

        fp = (
            cm[:, index].sum()
            - tp
        )

        fn = (
            cm[index, :].sum()
            - tp
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        if precision + recall > 0:
            f1 = (
                2.0
                * precision
                * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    precision_macro = float(
        np.mean(precision_values)
    )

    recall_macro = float(
        np.mean(recall_values)
    )

    f1_macro = float(
        np.mean(f1_values)
    )

    metrics = {
        "model": model_name,
        "test_rows": int(total_predictions),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "prediction_seconds": float(
            prediction_seconds
        ),
        "classes": label_encoder.classes_.tolist(),
        "random_seed": RANDOM_SEED,
        "prediction_batch_size": PREDICTION_BATCH_SIZE,
        "synthetic_data": False,
        "mock_data": False,
        "sampling": False,
        "rows_sampled": 0,
        "rows_evaluated": int(total_predictions),
    }

    return metrics, cm


def save_results(
    model,
    model_name,
    metrics,
    cm,
    class_names,
):
    if model_name == "RandomForestClassifier":
        model_file = OUTPUT_DIR / "model_rf.pkl"
        metrics_file = OUTPUT_DIR / "metrics_rf.json"
        cm_file = (
            OUTPUT_DIR
            / "confusion_matrix_rf.csv"
        )
    elif model_name == "XGBClassifier":
        model_file = OUTPUT_DIR / "model_xgb.pkl"
        metrics_file = OUTPUT_DIR / "metrics_xgb.json"
        cm_file = (
            OUTPUT_DIR
            / "confusion_matrix_xgb.csv"
        )
    else:
        fail(
            f"Unknown model name: {model_name}"
        )

    joblib.dump(
        model,
        model_file,
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    cm_df.index.name = "Actual"

    cm_df.to_csv(cm_file)

    if not model_file.exists():
        fail(
            f"Model artifact was not created: "
            f"{model_file}"
        )

    if model_file.stat().st_size == 0:
        fail(
            f"Model artifact is empty: "
            f"{model_file}"
        )


def validate_saved_confusion_matrix(
    path,
    expected_total,
    class_names,
):
    if not path.exists():
        fail(
            f"Missing confusion matrix: {path}"
        )

    cm_df = pd.read_csv(
        path,
        index_col=0,
    )

    if list(cm_df.index.astype(str)) != class_names:
        fail(
            f"Unexpected confusion matrix row labels "
            f"in {path}."
        )

    if list(cm_df.columns.astype(str)) != class_names:
        fail(
            f"Unexpected confusion matrix column labels "
            f"in {path}."
        )

    cm_values = cm_df.to_numpy(
        dtype=np.int64
    )

    if cm_values.sum() != expected_total:
        fail(
            f"Confusion matrix total mismatch: {path}\n"
            f"Expected: {expected_total:,}\n"
            f"Actual:   {cm_values.sum():,}"
        )

    if np.any(cm_values < 0):
        fail(
            f"Negative confusion matrix values detected: "
            f"{path}"
        )


def validate_artifacts(class_names):
    expected_files = [
        OUTPUT_DIR / "model_rf.pkl",
        OUTPUT_DIR / "model_xgb.pkl",
        OUTPUT_DIR / "metrics_rf.json",
        OUTPUT_DIR / "metrics_xgb.json",
        OUTPUT_DIR / "confusion_matrix_rf.csv",
        OUTPUT_DIR / "confusion_matrix_xgb.csv",
        OUTPUT_DIR / "label_mapping.json",
    ]

    for path in expected_files:
        if not path.exists():
            fail(
                f"Expected artifact missing: {path}"
            )

        if path.stat().st_size == 0:
            fail(
                f"Expected artifact is empty: {path}"
            )

    validate_saved_confusion_matrix(
        OUTPUT_DIR
        / "confusion_matrix_rf.csv",
        EXPECTED_TEST_ROWS,
        class_names,
    )

    validate_saved_confusion_matrix(
        OUTPUT_DIR
        / "confusion_matrix_xgb.csv",
        EXPECTED_TEST_ROWS,
        class_names,
    )

    with open(
        OUTPUT_DIR / "metrics_rf.json",
        "r",
        encoding="utf-8",
    ) as f:
        rf_metrics = json.load(f)

    with open(
        OUTPUT_DIR / "metrics_xgb.json",
        "r",
        encoding="utf-8",
    ) as f:
        xgb_metrics = json.load(f)

    for model_name, metrics in [
        ("Random Forest", rf_metrics),
        ("XGBoost", xgb_metrics),
    ]:
        if metrics.get("test_rows") != EXPECTED_TEST_ROWS:
            fail(
                f"{model_name} metrics contain an "
                "incorrect test row count."
            )

        if metrics.get("rows_evaluated") != EXPECTED_TEST_ROWS:
            fail(
                f"{model_name} metrics contain an "
                "incorrect rows_evaluated count."
            )

        if metrics.get("rows_sampled") != 0:
            fail(
                f"{model_name} reports sampled rows."
            )

        if metrics.get("sampling") is not False:
            fail(
                f"{model_name} sampling flag is not False."
            )

        if metrics.get("synthetic_data") is not False:
            fail(
                f"{model_name} synthetic_data flag is not False."
            )

        if metrics.get("mock_data") is not False:
            fail(
                f"{model_name} mock_data flag is not False."
            )

        for key in [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]:
            value = metrics.get(key)

            if value is None:
                fail(
                    f"{model_name} missing metric: {key}"
                )

            if not np.isfinite(float(value)):
                fail(
                    f"{model_name} metric is non-finite: "
                    f"{key}"
                )

    with open(
        OUTPUT_DIR / "label_mapping.json",
        "r",
        encoding="utf-8",
    ) as f:
        label_mapping = json.load(f)

    mapped_classes = [
        label_mapping[str(i)]
        for i in range(len(class_names))
    ]

    if mapped_classes != class_names:
        fail(
            "Saved label mapping does not match "
            "class ordering."
        )

    print()
    print(
        "    Artifact validation: PASSED"
    )


def train_random_forest(
    X_train,
    y_train,
    feature_columns,
    label_encoder,
    class_names,
):
    print()
    print(
        "[3/7] Training Random Forest baseline..."
    )

    rf_start = time.perf_counter()

    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=MODEL_N_JOBS,
    )

    rf_model.fit(
        X_train,
        y_train,
    )

    rf_training_seconds = (
        time.perf_counter()
        - rf_start
    )

    print(
        f"    Random Forest training completed "
        f"in {rf_training_seconds:.2f} seconds"
    )

    print()
    print(
        "[4/7] Evaluating Random Forest..."
    )

    rf_metrics, rf_cm = evaluate_model(
        rf_model,
        "RandomForestClassifier",
        feature_columns,
        label_encoder,
    )

    rf_metrics.update(
        {
            "training_seconds": float(
                rf_training_seconds
            ),
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": None,
            "class_weight": "balanced",
            "n_jobs": MODEL_N_JOBS,
            "data_source": (
                "CICIoT2023_Selected "
                "Experiment 002 processed data"
            ),
        }
    )

    save_results(
        rf_model,
        "RandomForestClassifier",
        rf_metrics,
        rf_cm,
        class_names,
    )

    print(
        f"    Accuracy:        "
        f"{rf_metrics['accuracy']:.6f}"
    )

    print(
        f"    Precision macro: "
        f"{rf_metrics['precision_macro']:.6f}"
    )

    print(
        f"    Recall macro:    "
        f"{rf_metrics['recall_macro']:.6f}"
    )

    print(
        f"    F1 macro:        "
        f"{rf_metrics['f1_macro']:.6f}"
    )

    print(
        f"    Test predictions: "
        f"{rf_metrics['test_rows']:,}"
    )

    print(
        f"    Confusion matrix total: "
        f"{rf_cm.sum():,}"
    )

    print(
        "    Random Forest artifacts saved."
    )

    release_memory(
        rf_model,
        rf_cm,
    )

    return rf_metrics


def train_xgboost(
    X_train,
    y_train,
    feature_columns,
    label_encoder,
    class_names,
):
    print()
    print(
        "[5/7] Training XGBoost baseline..."
    )

    xgb_start = time.perf_counter()

    xgb_model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        objective="multi:softprob",
        num_class=len(class_names),
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        n_jobs=MODEL_N_JOBS,
        tree_method="hist",
        max_bin=256,
    )

    xgb_model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    xgb_training_seconds = (
        time.perf_counter()
        - xgb_start
    )

    print(
        f"    XGBoost training completed "
        f"in {xgb_training_seconds:.2f} seconds"
    )

    print()
    print(
        "[6/7] Evaluating XGBoost..."
    )

    xgb_metrics, xgb_cm = evaluate_model(
        xgb_model,
        "XGBClassifier",
        feature_columns,
        label_encoder,
    )

    xgb_metrics.update(
        {
            "training_seconds": float(
                xgb_training_seconds
            ),
            "n_estimators": XGB_N_ESTIMATORS,
            "learning_rate": XGB_LEARNING_RATE,
            "max_depth": XGB_MAX_DEPTH,
            "tree_method": "hist",
            "n_jobs": MODEL_N_JOBS,
            "max_bin": 256,
            "data_source": (
                "CICIoT2023_Selected "
                "Experiment 002 processed data"
            ),
        }
    )

    save_results(
        xgb_model,
        "XGBClassifier",
        xgb_metrics,
        xgb_cm,
        class_names,
    )

    print(
        f"    Accuracy:        "
        f"{xgb_metrics['accuracy']:.6f}"
    )

    print(
        f"    Precision macro: "
        f"{xgb_metrics['precision_macro']:.6f}"
    )

    print(
        f"    Recall macro:    "
        f"{xgb_metrics['recall_macro']:.6f}"
    )

    print(
        f"    F1 macro:        "
        f"{xgb_metrics['f1_macro']:.6f}"
    )

    print(
        f"    Test predictions: "
        f"{xgb_metrics['test_rows']:,}"
    )

    print(
        f"    Confusion matrix total: "
        f"{xgb_cm.sum():,}"
    )

    print(
        "    XGBoost artifacts saved."
    )

    release_memory(
        xgb_model,
        xgb_cm,
    )

    return xgb_metrics


def main():
    print("=" * 70)
    print(
        "EXPERIMENT 003 - BASELINE MODEL TRAINING"
    )
    print("=" * 70)

    print()
    print("[0/7] Pre-flight validation...")

    verify_file(
        TRAIN_FILE,
        "training Parquet file",
    )

    verify_file(
        VAL_FILE,
        "validation Parquet file",
    )

    verify_file(
        TEST_FILE,
        "test Parquet file",
    )

    feature_columns = load_schema()

    print(
        f"    Features loaded from schema: "
        f"{len(feature_columns)}"
    )

    print(
        "    Feature count verified: YES"
    )

    print()
    print(
        "    Verifying Parquet metadata..."
    )

    verify_parquet_structure(
        TRAIN_FILE,
        EXPECTED_TRAIN_ROWS,
        "train",
        feature_columns,
    )

    verify_parquet_structure(
        VAL_FILE,
        EXPECTED_VAL_ROWS,
        "val",
        feature_columns,
    )

    verify_parquet_structure(
        TEST_FILE,
        EXPECTED_TEST_ROWS,
        "test",
        feature_columns,
    )

    total_rows = (
        EXPECTED_TRAIN_ROWS
        + EXPECTED_VAL_ROWS
        + EXPECTED_TEST_ROWS
    )

    if total_rows != EXPECTED_TOTAL_ROWS:
        fail(
            f"Expected total rows: "
            f"{EXPECTED_TOTAL_ROWS:,}\n"
            f"Calculated total: {total_rows:,}"
        )

    print(
        f"    Total rows: {total_rows:,}"
    )

    print(
        "    Parquet structure verified: YES"
    )

    class_names = validate_category_coverage()

    print()
    print(
        "    Dataset pre-flight validation: PASSED"
    )

    print()
    print(
        "[1/7] Preparing label encoder..."
    )

    label_encoder = LabelEncoder()

    label_encoder.fit(
        np.asarray(
            class_names,
            dtype=str,
        )
    )

    if list(label_encoder.classes_) != class_names:
        fail(
            "Label encoder class ordering is inconsistent."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_mapping = {
        str(index): label
        for index, label in enumerate(
            label_encoder.classes_
        )
    }

    with open(
        OUTPUT_DIR / "label_mapping.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            label_mapping,
            f,
            indent=4,
        )

    print(
        f"    Classes: {class_names}"
    )

    print(
        f"    Number of classes: "
        f"{len(class_names)}"
    )

    print()
    print(
        "[2/7] Loading complete training dataset..."
    )

    X_train, y_train = build_training_memmaps(
        feature_columns,
        label_encoder,
    )

    print()
    print(
        f"    Training matrix shape: "
        f"{X_train.shape}"
    )

    print(
        f"    Training matrix dtype: "
        f"{X_train.dtype}"
    )

    print(
        f"    Training labels shape: "
        f"{y_train.shape}"
    )

    rf_metrics = train_random_forest(
        X_train,
        y_train,
        feature_columns,
        label_encoder,
        class_names,
    )

    release_memory(
        X_train,
        y_train,
    )

    gc.collect()

    X_train, y_train = build_training_memmaps(
        feature_columns,
        label_encoder,
    )

    xgb_metrics = train_xgboost(
        X_train,
        y_train,
        feature_columns,
        label_encoder,
        class_names,
    )

    release_memory(
        X_train,
        y_train,
    )

    gc.collect()

    print()
    print(
        "[7/7] Final artifact verification..."
    )

    validate_artifacts(
        class_names
    )

    summary = {
        "experiment": "003_baseline_model_training",
        "dataset": "CICIoT2023_Selected",
        "train_rows": EXPECTED_TRAIN_ROWS,
        "validation_rows": EXPECTED_VAL_ROWS,
        "test_rows": EXPECTED_TEST_ROWS,
        "total_rows": EXPECTED_TOTAL_ROWS,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "classes": class_names,
        "synthetic_data": False,
        "mock_data": False,
        "sampling": False,
        "rows_sampled": 0,
        "rows_used_for_training": EXPECTED_TRAIN_ROWS,
        "rows_used_for_testing": EXPECTED_TEST_ROWS,
        "random_seed": RANDOM_SEED,
        "prediction_batch_size": PREDICTION_BATCH_SIZE,
        "parquet_batch_size": PARQUET_BATCH_SIZE,
        "disk_backed_training": True,
        "models": [
            "RandomForestClassifier",
            "XGBClassifier",
        ],
        "artifacts_verified": True,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
    }

    with open(
        OUTPUT_DIR / "experiment_003_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print("=" * 70)
    print(
        "EXPERIMENT 003 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)
    print(
        f"Train rows used: "
        f"{EXPECTED_TRAIN_ROWS:,}"
    )
    print(
        f"Test rows evaluated: "
        f"{EXPECTED_TEST_ROWS:,}"
    )
    print(
        f"Total dataset rows verified: "
        f"{EXPECTED_TOTAL_ROWS:,}"
    )
    print(
        f"Features: {EXPECTED_FEATURE_COUNT}"
    )
    print(
        f"Classes: {', '.join(class_names)}"
    )
    print(
        "Sampling: NO"
    )
    print(
        "Synthetic data: NO"
    )
    print(
        "Mock data: NO"
    )
    print(
        "Disk-backed batch loading: YES"
    )
    print(
        "Artifact verification: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()