import os
import sys
import json
import random
import hashlib
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

SEED = 42
EXPERIMENT_ID = "009"
EXPERIMENT_NAME = "hybrid_cnn_lstm"
DATA_DIR = Path("runs/experiment_002_preprocessing")
SCHEMA_PATH = Path("runs/experiment_001_dataset_char/schema_discovery_report.json")
OUTPUT_DIR = Path("runs/experiment_009_hybrid_cnn_lstm")
MODEL_PATH = OUTPUT_DIR / "best_model.keras"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
CONFIG_PATH = OUTPUT_DIR / "config.json"
TRAINING_CONFIG_PATH = OUTPUT_DIR / "training_config.json"
LABEL_MAPPING_PATH = OUTPUT_DIR / "label_mapping.json"
MODEL_SUMMARY_PATH = OUTPUT_DIR / "model_summary.txt"
VAL_METRICS_PATH = OUTPUT_DIR / "validation_metrics.json"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
VAL_CM_PATH = OUTPUT_DIR / "validation_confusion_matrix.csv"
TEST_CM_PATH = OUTPUT_DIR / "test_confusion_matrix.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.parquet"
ENVIRONMENT_PATH = OUTPUT_DIR / "environment.json"
RUN_LOG_PATH = OUTPUT_DIR / "run_log.json"
README_PATH = OUTPUT_DIR / "README.md"

TRAIN_PATH = DATA_DIR / "train_processed.parquet"
VAL_PATH = DATA_DIR / "val_processed.parquet"
TEST_PATH = DATA_DIR / "test_processed.parquet"

BATCH_SIZE = 1024
MAX_EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
CLIPNORM = 1.0
SHUFFLE_BUFFER_BATCHES = 32
PARQUET_BATCH_ROWS = 32768

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def validate_required_files():
    required = [TRAIN_PATH, VAL_PATH, TEST_PATH, SCHEMA_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Experiment 001/002 artifacts are missing:\n"
            + "\n".join(missing)
        )


def load_features():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    features = schema.get("features")

    if not isinstance(features, list) or not features:
        raise ValueError("No valid feature list found in schema_discovery_report.json")

    if len(features) != 39:
        raise ValueError(
            f"Expected 39 CICIoT2023 features from Experiment 001, found {len(features)}"
        )

    if len(set(features)) != len(features):
        raise ValueError("Duplicate feature names found in Experiment 001 schema")

    return features


def parquet_columns(path):
    return pq.ParquetFile(path).schema_arrow.names


def parquet_rows(path):
    return pq.ParquetFile(path).metadata.num_rows


def validate_parquet_schema(path, features):
    columns = parquet_columns(path)
    required = set(features + ["Category"])
    missing = sorted(required.difference(columns))

    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def collect_label_counts(path):
    counts = {}
    parquet = pq.ParquetFile(path)

    for batch in parquet.iter_batches(
        batch_size=PARQUET_BATCH_ROWS,
        columns=["Category"],
        use_threads=True
    ):
        values = batch.column(0).to_pylist()
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1

    return dict(sorted(counts.items()))


def validate_labels(train_counts, val_counts, test_counts):
    train_labels = set(train_counts)
    val_labels = set(val_counts)
    test_labels = set(test_counts)

    if not train_labels:
        raise ValueError("Training set contains no labels")

    unseen_val = sorted(val_labels.difference(train_labels))
    unseen_test = sorted(test_labels.difference(train_labels))

    if unseen_val:
        raise ValueError(f"Validation contains labels absent from training: {unseen_val}")

    if unseen_test:
        raise ValueError(f"Test contains labels absent from training: {unseen_test}")

    labels = sorted(train_labels)

    if labels != ["Benign", "DDoS", "Recon", "Web"]:
        raise ValueError(
            "Expected category labels ['Benign', 'DDoS', 'Recon', 'Web'], "
            f"found {labels}"
        )

    return labels


def compute_class_weights(train_counts, labels):
    total = float(sum(train_counts.values()))
    n_classes = float(len(labels))

    raw = {
        label: total / (n_classes * float(train_counts[label]))
        for label in labels
    }

    sqrt_weights = {
        label: float(np.sqrt(value))
        for label, value in raw.items()
    }

    mean_weight = float(np.mean(list(sqrt_weights.values())))

    normalized = {
        label: float(value / mean_weight)
        for label, value in sqrt_weights.items()
    }

    capped = {
        label: float(min(value, 5.0))
        for label, value in normalized.items()
    }

    return raw, normalized, capped


def parquet_batch_generator(
    path,
    features,
    label_to_index,
    class_weights=None,
    training=False
):
    parquet = pq.ParquetFile(path)
    columns = features + ["Category"]
    rng = np.random.default_rng(SEED)

    for batch in parquet.iter_batches(
        batch_size=PARQUET_BATCH_ROWS,
        columns=columns,
        use_threads=True
    ):
        frame = batch.to_pandas()

        x = frame[features].to_numpy(dtype=np.float32, copy=True)

        if not np.isfinite(x).all():
            bad = int(np.size(x) - np.isfinite(x).sum())
            raise ValueError(
                f"Non-finite values encountered in {path}: {bad}"
            )

        labels = frame["Category"].astype(str).to_numpy()
        y = np.fromiter(
            (label_to_index[label] for label in labels),
            dtype=np.int32,
            count=len(labels)
        )

        if training:
            order = rng.permutation(len(x))
            x = x[order]
            y = y[order]
            labels = labels[order]

        for start in range(0, len(x), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(x))

            xb = x[start:end].reshape(-1, len(features), 1)
            yb = y[start:end]

            if class_weights is None:
                yield xb, yb
            else:
                wb = np.fromiter(
                    (
                        class_weights[
                            labels[i]
                        ]
                        for i in range(start, end)
                    ),
                    dtype=np.float32,
                    count=end - start
                )
                yield xb, yb, wb


def make_dataset(
    path,
    features,
    label_to_index,
    class_weights=None,
    training=False
):
    if class_weights is None:
        signature = (
            tf.TensorSpec(
                shape=(None, len(features), 1),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(None,),
                dtype=tf.int32
            )
        )
    else:
        signature = (
            tf.TensorSpec(
                shape=(None, len(features), 1),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(None,),
                dtype=tf.int32
            ),
            tf.TensorSpec(
                shape=(None,),
                dtype=tf.float32
            )
        )

    dataset = tf.data.Dataset.from_generator(
        lambda: parquet_batch_generator(
            path,
            features,
            label_to_index,
            class_weights=class_weights,
            training=training
        ),
        output_signature=signature
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(n_features, n_classes):
    inputs = tf.keras.Input(
        shape=(n_features, 1),
        name="feature_vector"
    )

    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1d_1"
    )(inputs)

    x = tf.keras.layers.BatchNormalization(
        name="batch_norm_1"
    )(x)

    x = tf.keras.layers.MaxPooling1D(
        pool_size=2,
        name="max_pool_1"
    )(x)

    x = tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1d_2"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_norm_2"
    )(x)

    x = tf.keras.layers.MaxPooling1D(
        pool_size=2,
        name="max_pool_2"
    )(x)

    x = tf.keras.layers.LSTM(
        units=64,
        return_sequences=False,
        name="lstm"
    )(x)

    x = tf.keras.layers.Dense(
        units=128,
        activation="relu",
        name="dense_1"
    )(x)

    x = tf.keras.layers.Dropout(
        rate=0.30,
        name="dropout"
    )(x)

    outputs = tf.keras.layers.Dense(
        units=n_classes,
        activation="softmax",
        name="category"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Hybrid_CNN_LSTM_FeatureAxis"
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIPNORM
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )

    return model


def save_model_summary(model):
    lines = []
    model.summary(print_fn=lines.append)

    with open(MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def evaluate_model(
    model,
    path,
    features,
    labels,
    label_to_index,
    metrics_path,
    cm_path,
    predictions_path=None
):
    parquet = pq.ParquetFile(path)

    y_true_parts = []
    y_pred_parts = []
    confidence_parts = []

    row_offset = 0
    prediction_frames = []

    for batch in parquet.iter_batches(
        batch_size=PARQUET_BATCH_ROWS,
        columns=features + ["Category"],
        use_threads=True
    ):
        frame = batch.to_pandas()

        x = frame[features].to_numpy(dtype=np.float32, copy=True)

        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite values encountered during evaluation of {path}")

        true_labels = frame["Category"].astype(str).to_numpy()

        y_true = np.fromiter(
            (label_to_index[label] for label in true_labels),
            dtype=np.int32,
            count=len(true_labels)
        )

        x = x.reshape(-1, len(features), 1)

        probabilities = model.predict(
            x,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        y_pred = np.argmax(probabilities, axis=1).astype(np.int32)
        confidence = np.max(probabilities, axis=1).astype(np.float32)

        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)
        confidence_parts.append(confidence)

        if predictions_path is not None:
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_index": np.arange(
                            row_offset,
                            row_offset + len(y_true),
                            dtype=np.int64
                        ),
                        "true_index": y_true,
                        "true_label": [
                            labels[i] for i in y_true
                        ],
                        "predicted_index": y_pred,
                        "predicted_label": [
                            labels[i] for i in y_pred
                        ],
                        "confidence": confidence
                    }
                )
            )

        row_offset += len(y_true)

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)

    expected_rows = parquet.metadata.num_rows

    if len(y_true) != expected_rows:
        raise RuntimeError(
            f"Evaluation row mismatch: expected {expected_rows}, got {len(y_true)}"
        )

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(len(labels)),
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(len(labels)),
            average="weighted",
            zero_division=0
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(labels))
    )

    if int(cm.sum()) != expected_rows:
        raise RuntimeError(
            f"Confusion matrix total {cm.sum()} does not equal expected rows {expected_rows}"
        )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
    cm_df.index.name = "actual"
    cm_df.columns.name = "predicted"
    cm_df.to_csv(cm_path)

    metrics = {
        "rows_evaluated": int(expected_rows),
        "accuracy": float(accuracy),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "f1_weighted": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix_total": int(cm.sum()),
        "integrity_check_passed": bool(cm.sum() == expected_rows)
    }

    write_json(metrics_path, metrics)

    if predictions_path is not None:
        predictions = pd.concat(
            prediction_frames,
            ignore_index=True
        )

        if len(predictions) != expected_rows:
            raise RuntimeError(
                "Prediction artifact row count does not match source test rows"
            )

        predictions.to_parquet(
            predictions_path,
            index=False
        )

    return metrics


validate_required_files()

features = load_features()

for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    validate_parquet_schema(path, features)

train_rows = parquet_rows(TRAIN_PATH)
val_rows = parquet_rows(VAL_PATH)
test_rows = parquet_rows(TEST_PATH)

expected_rows = {
    "train": 7702608,
    "validation": 1236682,
    "test": 1773902
}

actual_rows = {
    "train": train_rows,
    "validation": val_rows,
    "test": test_rows
}

if actual_rows != expected_rows:
    raise ValueError(
        f"Dataset row counts do not match verified Experiments 001/002. "
        f"Expected {expected_rows}, found {actual_rows}"
    )

train_counts = collect_label_counts(TRAIN_PATH)
val_counts = collect_label_counts(VAL_PATH)
test_counts = collect_label_counts(TEST_PATH)

labels = validate_labels(
    train_counts,
    val_counts,
    test_counts
)

label_to_index = {
    label: index
    for index, label in enumerate(labels)
}

index_to_label = {
    str(index): label
    for label, index in label_to_index.items()
}

raw_class_weights, normalized_class_weights, applied_class_weights = (
    compute_class_weights(
        train_counts,
        labels
    )
)

write_json(
    LABEL_MAPPING_PATH,
    {
        "label_to_index": label_to_index,
        "index_to_label": index_to_label
    }
)

environment = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "tensorflow_version": tf.__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "scikit_learn_version": sklearn.__version__,
    "gpu_devices": [
        device.name
        for device in tf.config.list_physical_devices("GPU")
    ]
}

write_json(ENVIRONMENT_PATH, environment)

training_config = {
    "experiment_id": EXPERIMENT_ID,
    "experiment_name": EXPERIMENT_NAME,
    "seed": SEED,
    "deterministic_ops_requested": True,
    "dataset": "CICIoT2023_Selected",
    "dataset_representation": "official_CICIoT2023_CSV_feature_observations",
    "synthetic_data": False,
    "mock_data": False,
    "generated_training_rows": False,
    "temporal_sequence_claim": False,
    "input_semantics": "ordered_39_feature_axis",
    "lstm_semantics": "dependencies_across_convolutionally_transformed_feature_positions",
    "explicit_limitation": "The released CSV representation does not retain an explicit timestamp. The LSTM therefore does not represent verified chronological sequences of consecutive CSV rows.",
    "feature_count": len(features),
    "features": features,
    "input_shape": [len(features), 1],
    "classes": labels,
    "class_count": len(labels),
    "row_counts": actual_rows,
    "class_distributions": {
        "train": train_counts,
        "validation": val_counts,
        "test": test_counts
    },
    "class_weight_method": "sqrt_inverse_frequency_normalized_then_capped",
    "raw_inverse_frequency_weights": raw_class_weights,
    "normalized_sqrt_weights": normalized_class_weights,
    "applied_class_weights": applied_class_weights,
    "class_weight_cap": 5.0,
    "batch_size": BATCH_SIZE,
    "parquet_batch_rows": PARQUET_BATCH_ROWS,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "learning_rate": LEARNING_RATE,
    "clipnorm": CLIPNORM,
    "optimizer": "Adam",
    "loss": "SparseCategoricalCrossentropy",
    "checkpoint_monitor": "val_loss",
    "checkpoint_mode": "min",
    "restore_best_weights": True
}

write_json(TRAINING_CONFIG_PATH, training_config)

config = {
    "experiment_id": EXPERIMENT_ID,
    "status": "initialized",
    "created_utc": utc_now(),
    "input_artifacts": {
        "train_processed": str(TRAIN_PATH),
        "validation_processed": str(VAL_PATH),
        "test_processed": str(TEST_PATH),
        "schema_report": str(SCHEMA_PATH)
    },
    "input_sha256": {
        "schema_report": sha256_file(SCHEMA_PATH)
    },
    "output_directory": str(OUTPUT_DIR),
    "model_type": "Hybrid CNN-LSTM",
    "model_interpretation": "non-temporal feature-axis hybrid neural classifier"
}

write_json(CONFIG_PATH, config)

run_log = {
    "experiment_id": EXPERIMENT_ID,
    "started_utc": utc_now(),
    "completed_utc": None,
    "status": "running",
    "error": None
}

write_json(RUN_LOG_PATH, run_log)

try:
    train_dataset = make_dataset(
        TRAIN_PATH,
        features,
        label_to_index,
        class_weights=applied_class_weights,
        training=True
    )

    val_dataset = make_dataset(
        VAL_PATH,
        features,
        label_to_index,
        class_weights=None,
        training=False
    )

    model = build_model(
        len(features),
        len(labels)
    )

    save_model_summary(model)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    training_start = time.perf_counter()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    training_elapsed = time.perf_counter() - training_start

    history_df = pd.DataFrame(history.history)
    history_df.insert(
        0,
        "epoch",
        np.arange(1, len(history_df) + 1)
    )
    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    if not MODEL_PATH.exists():
        raise RuntimeError("Best-model checkpoint was not created")

    best_model = tf.keras.models.load_model(
        MODEL_PATH
    )

    expected_input_shape = (None, len(features), 1)

    if tuple(best_model.input_shape) != expected_input_shape:
        raise RuntimeError(
            f"Saved model input shape {best_model.input_shape} does not match "
            f"expected {expected_input_shape}"
        )

    validation_metrics = evaluate_model(
        best_model,
        VAL_PATH,
        features,
        labels,
        label_to_index,
        VAL_METRICS_PATH,
        VAL_CM_PATH
    )

    test_metrics = evaluate_model(
        best_model,
        TEST_PATH,
        features,
        labels,
        label_to_index,
        TEST_METRICS_PATH,
        TEST_CM_PATH,
        predictions_path=TEST_PREDICTIONS_PATH
    )

    config["status"] = "completed"
    config["completed_utc"] = utc_now()
    config["epochs_completed"] = len(history_df)
    config["training_elapsed_seconds"] = float(training_elapsed)
    config["best_model_path"] = str(MODEL_PATH)
    config["best_model_sha256"] = sha256_file(MODEL_PATH)
    config["validation_metrics_path"] = str(VAL_METRICS_PATH)
    config["test_metrics_path"] = str(TEST_METRICS_PATH)
    config["validation_confusion_matrix_path"] = str(VAL_CM_PATH)
    config["test_confusion_matrix_path"] = str(TEST_CM_PATH)
    config["test_predictions_path"] = str(TEST_PREDICTIONS_PATH)
    config["validation_accuracy"] = validation_metrics["accuracy"]
    config["validation_f1_macro"] = validation_metrics["f1_macro"]
    config["test_accuracy"] = test_metrics["accuracy"]
    config["test_f1_macro"] = test_metrics["f1_macro"]

    write_json(CONFIG_PATH, config)

    readme = f"""# Experiment 009 — Hybrid CNN-LSTM

Status: completed

Dataset: selected official CICIoT2023 CSV feature files

Training rows: {train_rows}
Validation rows: {val_rows}
Test rows: {test_rows}

Features: {len(features)}
Classes: {", ".join(labels)}

Model: Conv1D → BatchNormalization → MaxPooling1D → Conv1D → BatchNormalization → MaxPooling1D → LSTM → Dense → Dropout → Softmax

Input shape: ({len(features)}, 1)

The model operates on the ordered CICIoT2023 feature axis. The LSTM is used to model dependencies across convolutionally transformed feature positions. It is not interpreted as processing verified chronological sequences of consecutive CSV rows because the released CSV representation does not retain an explicit timestamp.

No synthetic training data, mock observations, generated metrics, generated confusion matrices, or fabricated performance values are used.

All reported validation and test metrics are calculated from inference using the saved best checkpoint.

Validation accuracy: {validation_metrics["accuracy"]}
Validation macro F1: {validation_metrics["f1_macro"]}
Test accuracy: {test_metrics["accuracy"]}
Test macro F1: {test_metrics["f1_macro"]}
"""

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(readme)

    run_log["status"] = "completed"
    run_log["completed_utc"] = utc_now()
    write_json(RUN_LOG_PATH, run_log)

    print(json.dumps({
        "status": "completed",
        "experiment": EXPERIMENT_ID,
        "model": str(MODEL_PATH),
        "epochs_completed": len(history_df),
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["f1_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["f1_macro"]
    }, indent=2))

except Exception as exc:
    run_log["status"] = "failed"
    run_log["completed_utc"] = utc_now()
    run_log["error"] = {
        "type": type(exc).__name__,
        "message": str(exc)
    }
    write_json(RUN_LOG_PATH, run_log)

    config["status"] = "failed"
    config["completed_utc"] = utc_now()
    config["error"] = {
        "type": type(exc).__name__,
        "message": str(exc)
    }
    write_json(CONFIG_PATH, config)

    raise