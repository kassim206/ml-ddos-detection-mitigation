import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("runs/experiment_002_preprocessing")
EXP001_DIR = Path("runs/experiment_001_dataset_char")
EXP009_DIR = Path("runs/experiment_009_hybrid_cnn_lstm")

MODEL_PATH = EXP009_DIR / "best_model.keras"
VAL_PATH = DATA_DIR / "val_processed.parquet"
TEST_PATH = DATA_DIR / "test_processed.parquet"

LABELS = ["Benign", "DDoS", "Recon", "Web"]
LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}

EXPECTED_ROWS = {
    "validation": 1236682,
    "test": 1773902
}

BATCH_SIZE = 1024
PARQUET_BATCH_SIZE = 32768


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_features():
    schema_path = EXP001_DIR / "schema_discovery_report.json"

    if not schema_path.exists():
        raise FileNotFoundError(schema_path)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    features = schema["features"]

    if len(features) != 39:
        raise ValueError(
            f"Expected exactly 39 features, found {len(features)}"
        )

    if len(set(features)) != 39:
        raise ValueError("Duplicate feature names detected")

    return features


X_COLUMNS = load_features()


def validate_artifacts():
    required = [
        MODEL_PATH,
        VAL_PATH,
        TEST_PATH
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required artifacts:\n"
            + "\n".join(missing)
        )

    val_rows = pq.ParquetFile(
        VAL_PATH
    ).metadata.num_rows

    test_rows = pq.ParquetFile(
        TEST_PATH
    ).metadata.num_rows

    if val_rows != EXPECTED_ROWS["validation"]:
        raise ValueError(
            f"Validation row count mismatch: "
            f"{val_rows} != {EXPECTED_ROWS['validation']}"
        )

    if test_rows != EXPECTED_ROWS["test"]:
        raise ValueError(
            f"Test row count mismatch: "
            f"{test_rows} != {EXPECTED_ROWS['test']}"
        )

    return val_rows, test_rows


def inspect_model(model):
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)

    expected_input = (
        None,
        len(X_COLUMNS),
        1
    )

    expected_output = (
        None,
        len(LABELS)
    )

    if input_shape != expected_input:
        raise ValueError(
            f"Unexpected model input shape: {input_shape}. "
            f"Expected {expected_input}"
        )

    if output_shape != expected_output:
        raise ValueError(
            f"Unexpected model output shape: {output_shape}. "
            f"Expected {expected_output}"
        )


def evaluate_split(model, split_name, parquet_path):
    parquet_file = pq.ParquetFile(parquet_path)

    expected_rows = EXPECTED_ROWS[split_name]

    true_indices = np.empty(
        expected_rows,
        dtype=np.int8
    )

    predicted_indices = np.empty(
        expected_rows,
        dtype=np.int8
    )

    confidences = np.empty(
        expected_rows,
        dtype=np.float32
    )

    offset = 0
    inference_seconds = 0.0

    columns = X_COLUMNS + ["Category"]

    for batch in parquet_file.iter_batches(
        batch_size=PARQUET_BATCH_SIZE,
        columns=columns
    ):
        df = batch.to_pandas()

        features = df[
            X_COLUMNS
        ].to_numpy(
            dtype=np.float32,
            copy=True
        )

        if not np.isfinite(features).all():
            raise ValueError(
                f"Non-finite feature values detected in {split_name}"
            )

        labels = df[
            "Category"
        ].astype(str).to_numpy()

        unknown = sorted(
            set(labels).difference(LABEL_TO_INDEX)
        )

        if unknown:
            raise ValueError(
                f"Unknown labels in {split_name}: {unknown}"
            )

        y_true = np.fromiter(
            (
                LABEL_TO_INDEX[label]
                for label in labels
            ),
            dtype=np.int8,
            count=len(labels)
        )

        x_model = features.reshape(
            -1,
            len(X_COLUMNS),
            1
        )

        start = time.perf_counter()

        probabilities = model.predict(
            x_model,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        inference_seconds += (
            time.perf_counter() - start
        )

        if probabilities.shape != (
            len(df),
            len(LABELS)
        ):
            raise ValueError(
                f"Unexpected prediction shape in {split_name}: "
                f"{probabilities.shape}"
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"Non-finite probabilities detected in {split_name}"
            )

        y_pred = np.argmax(
            probabilities,
            axis=1
        ).astype(np.int8)

        confidence = np.max(
            probabilities,
            axis=1
        ).astype(np.float32)

        end = offset + len(df)

        true_indices[offset:end] = y_true
        predicted_indices[offset:end] = y_pred
        confidences[offset:end] = confidence

        offset = end

        print(
            f"[{split_name.upper()}] "
            f"{offset:,}/{expected_rows:,}"
        )

    if offset != expected_rows:
        raise ValueError(
            f"{split_name} evaluation produced "
            f"{offset} rows; expected {expected_rows}"
        )

    cm = confusion_matrix(
        true_indices,
        predicted_indices,
        labels=np.arange(len(LABELS))
    )

    if int(cm.sum()) != expected_rows:
        raise ValueError(
            f"{split_name} confusion matrix total mismatch"
        )

    accuracy = accuracy_score(
        true_indices,
        predicted_indices
    )

    precision_macro = precision_score(
        true_indices,
        predicted_indices,
        labels=np.arange(len(LABELS)),
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        true_indices,
        predicted_indices,
        labels=np.arange(len(LABELS)),
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        true_indices,
        predicted_indices,
        labels=np.arange(len(LABELS)),
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        true_indices,
        predicted_indices,
        labels=np.arange(len(LABELS)),
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        true_indices,
        predicted_indices,
        labels=np.arange(len(LABELS)),
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        true_indices,
        predicted_indices,
        labels=np.arange(len(LABELS)),
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        true_indices,
        predicted_indices,
        labels=np.arange(len(LABELS)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "split": split_name,
        "rows_evaluated": int(expected_rows),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "inference_seconds": float(inference_seconds),
        "classification_report": report
    }

    return (
        metrics,
        cm,
        true_indices,
        predicted_indices,
        confidences
    )


def save_confusion_matrix(cm, path):
    df = pd.DataFrame(
        cm,
        index=LABELS,
        columns=LABELS
    )

    df.index.name = "actual"
    df.columns.name = "predicted"

    df.to_csv(path)


def save_predictions(
    true_indices,
    predicted_indices,
    confidences,
    path
):
    rows = len(true_indices)

    df = pd.DataFrame({
        "row_index": np.arange(
            rows,
            dtype=np.int64
        ),
        "true_index": true_indices,
        "true_label": [
            LABELS[index]
            for index in true_indices
        ],
        "predicted_index": predicted_indices,
        "predicted_label": [
            LABELS[index]
            for index in predicted_indices
        ],
        "confidence": confidences
    })

    df.to_parquet(
        path,
        index=False
    )


def main():
    started = datetime.now(
        timezone.utc
    ).isoformat()

    print("[START] Experiment 009 evaluation only")
    print("[INFO] No training will be performed")

    val_rows, test_rows = validate_artifacts()

    model_sha256_before = sha256_file(
        MODEL_PATH
    )

    print(
        f"[MODEL] {MODEL_PATH}"
    )

    print(
        f"[MODEL SHA256] {model_sha256_before}"
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    inspect_model(model)

    print(
        f"[MODEL INPUT] {model.input_shape}"
    )

    print(
        f"[MODEL OUTPUT] {model.output_shape}"
    )

    print(
        f"[VALIDATION ROWS] {val_rows:,}"
    )

    validation = evaluate_split(
        model,
        "validation",
        VAL_PATH
    )

    (
        validation_metrics,
        validation_cm,
        _,
        _,
        _
    ) = validation

    with open(
        EXP009_DIR / "validation_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            validation_metrics,
            f,
            indent=2
        )

    save_confusion_matrix(
        validation_cm,
        EXP009_DIR / "validation_confusion_matrix.csv"
    )

    print(
        "[VALIDATION] "
        f"accuracy={validation_metrics['accuracy']:.6f} "
        f"macro_f1={validation_metrics['f1_macro']:.6f}"
    )

    print(
        f"[TEST ROWS] {test_rows:,}"
    )

    test = evaluate_split(
        model,
        "test",
        TEST_PATH
    )

    (
        test_metrics,
        test_cm,
        test_true,
        test_predicted,
        test_confidences
    ) = test

    with open(
        EXP009_DIR / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            test_metrics,
            f,
            indent=2
        )

    save_confusion_matrix(
        test_cm,
        EXP009_DIR / "test_confusion_matrix.csv"
    )

    save_predictions(
        test_true,
        test_predicted,
        test_confidences,
        EXP009_DIR / "test_predictions.parquet"
    )

    model_sha256_after = sha256_file(
        MODEL_PATH
    )

    if model_sha256_before != model_sha256_after:
        raise RuntimeError(
            "Model file changed during evaluation"
        )

    evaluation_log = {
        "experiment_id": "009",
        "evaluation_only": True,
        "training_performed": False,
        "model_path": str(MODEL_PATH),
        "model_sha256": model_sha256_after,
        "selected_checkpoint_epoch": 4,
        "selected_checkpoint_monitor": "val_loss",
        "completed_training_epochs": 7,
        "partial_epoch_excluded": 8,
        "feature_count": len(X_COLUMNS),
        "input_shape": [
            39,
            1
        ],
        "labels": LABELS,
        "validation_rows": val_rows,
        "test_rows": test_rows,
        "batch_size": BATCH_SIZE,
        "synthetic_results": False,
        "mock_results": False,
        "prediction_generation": "direct_best_checkpoint_inference",
        "started_utc": started,
        "completed_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__
        }
    }

    with open(
        EXP009_DIR / "evaluation_log.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            evaluation_log,
            f,
            indent=2
        )

    print()
    print("[TEST RESULTS]")
    print(
        f"Accuracy:        {test_metrics['accuracy']:.6f}"
    )
    print(
        f"Macro Precision: {test_metrics['precision_macro']:.6f}"
    )
    print(
        f"Macro Recall:    {test_metrics['recall_macro']:.6f}"
    )
    print(
        f"Macro F1:        {test_metrics['f1_macro']:.6f}"
    )
    print(
        f"Weighted F1:     {test_metrics['f1_weighted']:.6f}"
    )

    print()
    print("[TEST CONFUSION MATRIX]")
    print(
        pd.DataFrame(
            test_cm,
            index=LABELS,
            columns=LABELS
        ).to_string()
    )

    print()
    print("[PER-CLASS TEST METRICS]")

    for label in LABELS:
        row = test_metrics[
            "classification_report"
        ][label]

        print(
            f"{label:8s} "
            f"precision={row['precision']:.6f} "
            f"recall={row['recall']:.6f} "
            f"f1={row['f1-score']:.6f} "
            f"support={int(row['support']):,}"
        )

    print()
    print("[COMPLETE] Evaluation finished successfully")
    print(
        f"[VERIFIED MODEL SHA256] {model_sha256_after}"
    )


if __name__ == "__main__":
    main()
