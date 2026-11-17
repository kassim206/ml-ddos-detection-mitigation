import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]

EXP_DIR = ROOT / "runs" / "experiment_004_cnn_dense"

VAL_PATH = (
    ROOT
    / "runs"
    / "experiment_002_preprocessing"
    / "val_processed.parquet"
)

MODEL_PATH = EXP_DIR / "best_model.keras"
CONFIG_PATH = EXP_DIR / "training_config.json"
LABEL_PATH = EXP_DIR / "label_mapping.json"

DIAG_DIR = EXP_DIR / "diagnostics_best_checkpoint"

EXPECTED_VAL_ROWS = 1_236_682
EXPECTED_FEATURES = 39
BATCH_SIZE = 8192

print("=" * 70)
print("EXPERIMENT 004 — CNN-DENSE BEST CHECKPOINT DIAGNOSTIC")
print("=" * 70)

print("\nREAD-ONLY MODEL EVALUATION")
print("No training will be performed.")
print("The saved model will not be modified.")

required_files = [
    MODEL_PATH,
    CONFIG_PATH,
    LABEL_PATH,
    VAL_PATH,
]

print("\nChecking required files...")

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )
    print(f"PASS: {path}")

DIAG_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading Experiment 004 configuration...")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_mapping_raw = json.load(f)

if "features" not in config:
    raise RuntimeError(
        "training_config.json does not contain a 'features' field."
    )

features = config["features"]

if len(features) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Feature-count mismatch.\n"
        f"Expected: {EXPECTED_FEATURES}\n"
        f"Found: {len(features)}"
    )

print(f"Feature count: {len(features)} — PASS")

expected_mapping = {
    "Benign": 0,
    "DDoS": 1,
    "Recon": 2,
    "Web": 3,
}


def normalize_label_mapping(raw_mapping):
    if all(
        class_name in raw_mapping
        for class_name in expected_mapping
    ):
        return {
            str(k): int(v)
            for k, v in raw_mapping.items()
        }

    if "class_to_index" in raw_mapping:
        return {
            str(k): int(v)
            for k, v in raw_mapping["class_to_index"].items()
        }

    raise RuntimeError(
        "Unrecognized label_mapping.json structure.\n"
        f"Contents:\n{raw_mapping}"
    )


mapping = normalize_label_mapping(label_mapping_raw)

if mapping != expected_mapping:
    raise RuntimeError(
        "Label mapping mismatch.\n"
        f"Expected: {expected_mapping}\n"
        f"Found:    {mapping}"
    )

index_to_class = {
    index: class_name
    for class_name, index in mapping.items()
}

class_names = [
    index_to_class[i]
    for i in range(len(index_to_class))
]

print("Label mapping: PASS")

for name in class_names:
    print(f"  {name:10s} -> {mapping[name]}")

print("\nOpening validation Parquet...")

parquet_file = pq.ParquetFile(VAL_PATH)

val_rows = parquet_file.metadata.num_rows
val_columns = parquet_file.schema.names

print(f"Validation rows: {val_rows:,}")

if val_rows != EXPECTED_VAL_ROWS:
    raise RuntimeError(
        "Validation row-count mismatch.\n"
        f"Expected: {EXPECTED_VAL_ROWS:,}\n"
        f"Found:    {val_rows:,}"
    )

print("Validation row count: PASS")

required_columns = features + ["Category"]

missing_columns = [
    column
    for column in required_columns
    if column not in val_columns
]

if missing_columns:
    raise RuntimeError(
        "Validation dataset is missing required columns:\n"
        f"{missing_columns}"
    )

print("Validation columns: PASS")

print("\nLoading saved best checkpoint...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print(f"Model name:   {model.name}")
print(f"Input shape:  {model.input_shape}")
print(f"Output shape: {model.output_shape}")

actual_input_shape = tuple(model.input_shape[1:])

if actual_input_shape != (EXPECTED_FEATURES, 1):
    raise RuntimeError(
        "Unexpected model input shape.\n"
        f"Expected: (None, {EXPECTED_FEATURES}, 1)\n"
        f"Found:    {model.input_shape}"
    )

if model.output_shape[-1] != len(class_names):
    raise RuntimeError(
        "Unexpected model output size.\n"
        f"Expected classes: {len(class_names)}\n"
        f"Output shape:     {model.output_shape}"
    )

print("Model input/output validation: PASS")

print("\nMODEL SUMMARY")
print("-" * 70)

model.summary()

print("\n" + "=" * 70)
print("RUNNING VALIDATION INFERENCE")
print("=" * 70)

print(f"Rows:       {EXPECTED_VAL_ROWS:,}")
print(f"Batch size: {BATCH_SIZE:,}")
print("\nInference only.\n")

y_true_parts = []
y_pred_parts = []

rows_processed = 0

for batch_number, batch in enumerate(
    parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=required_columns,
    ),
    start=1,
):
    df = batch.to_pandas()

    X = df[features].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if np.isnan(X).any():
        raise RuntimeError(
            f"NaN detected in validation batch {batch_number}."
        )

    if np.isinf(X).any():
        raise RuntimeError(
            f"Infinity detected in validation batch {batch_number}."
        )

    labels = df["Category"].astype(str)

    unknown_labels = (
        set(labels.unique())
        - set(mapping.keys())
    )

    if unknown_labels:
        raise RuntimeError(
            "Unexpected labels found in validation data:\n"
            f"{sorted(unknown_labels)}"
        )

    y_true_batch = (
        labels
        .map(mapping)
        .to_numpy(dtype=np.int64)
    )

    X = X.reshape(
        -1,
        EXPECTED_FEATURES,
        1,
    )

    probabilities = model(
        X,
        training=False,
    ).numpy()

    if probabilities.shape[1] != len(class_names):
        raise RuntimeError(
            "Prediction output has unexpected shape:\n"
            f"{probabilities.shape}"
        )

    y_pred_batch = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

    y_true_parts.append(y_true_batch)
    y_pred_parts.append(y_pred_batch)

    rows_processed += len(df)

    if (
        batch_number == 1
        or batch_number % 20 == 0
        or rows_processed == EXPECTED_VAL_ROWS
    ):
        print(
            f"Processed "
            f"{rows_processed:,} / "
            f"{EXPECTED_VAL_ROWS:,}"
        )

print("\nCombining predictions...")

y_true = np.concatenate(y_true_parts)
y_pred = np.concatenate(y_pred_parts)

if len(y_true) != EXPECTED_VAL_ROWS:
    raise RuntimeError(
        "True-label count mismatch after inference.\n"
        f"Expected: {EXPECTED_VAL_ROWS:,}\n"
        f"Found:    {len(y_true):,}"
    )

if len(y_pred) != EXPECTED_VAL_ROWS:
    raise RuntimeError(
        "Prediction count mismatch after inference.\n"
        f"Expected: {EXPECTED_VAL_ROWS:,}\n"
        f"Found:    {len(y_pred):,}"
    )

print(
    f"Inference complete: "
    f"{len(y_pred):,} predictions — PASS"
)

true_counts = np.bincount(
    y_true,
    minlength=len(class_names),
)

pred_counts = np.bincount(
    y_pred,
    minlength=len(class_names),
)

print("\n" + "=" * 70)
print("TRUE CLASS DISTRIBUTION")
print("=" * 70)

for i, name in enumerate(class_names):
    percentage = (
        true_counts[i]
        / len(y_true)
        * 100.0
    )

    print(
        f"{name:10s}: "
        f"{true_counts[i]:>10,} "
        f"({percentage:8.4f}%)"
    )

print("\n" + "=" * 70)
print("PREDICTED CLASS DISTRIBUTION")
print("=" * 70)

for i, name in enumerate(class_names):
    percentage = (
        pred_counts[i]
        / len(y_pred)
        * 100.0
    )

    print(
        f"{name:10s}: "
        f"{pred_counts[i]:>10,} "
        f"({percentage:8.4f}%)"
    )

accuracy = accuracy_score(
    y_true,
    y_pred,
)

(
    macro_precision,
    macro_recall,
    macro_f1,
    _,
) = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

(
    weighted_precision,
    weighted_recall,
    weighted_f1,
    _,
) = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
)

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"Accuracy:           {accuracy:.10f}")
print(f"Macro Precision:    {macro_precision:.10f}")
print(f"Macro Recall:       {macro_recall:.10f}")
print(f"Macro F1:           {macro_f1:.10f}")
print(f"Weighted Precision: {weighted_precision:.10f}")
print(f"Weighted Recall:    {weighted_recall:.10f}")
print(f"Weighted F1:        {weighted_f1:.10f}")

print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

for name in class_names:
    class_result = report[name]

    print(
        f"{name:10s} "
        f"P={class_result['precision']:.6f} "
        f"R={class_result['recall']:.6f} "
        f"F1={class_result['f1-score']:.6f} "
        f"Support={int(class_result['support']):,}"
    )

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("Rows    = Actual")
print("Columns = Predicted")
print()
print("Class order:", class_names)
print()
print(cm)

cm_total = int(cm.sum())
true_total = int(true_counts.sum())
pred_total = int(pred_counts.sum())

if cm_total != EXPECTED_VAL_ROWS:
    raise RuntimeError(
        "Confusion matrix total does not equal expected validation rows."
    )

if true_total != EXPECTED_VAL_ROWS:
    raise RuntimeError(
        "True distribution total does not equal expected validation rows."
    )

if pred_total != EXPECTED_VAL_ROWS:
    raise RuntimeError(
        "Predicted distribution total does not equal expected validation rows."
    )

metrics = {
    "experiment": "004",
    "diagnostic": "best_checkpoint_validation",
    "model_file": str(MODEL_PATH),
    "validation_file": str(VAL_PATH),
    "model_name": model.name,
    "model_input_shape": str(model.input_shape),
    "model_output_shape": str(model.output_shape),
    "validation_rows": int(len(y_true)),
    "batch_size": BATCH_SIZE,
    "accuracy": float(accuracy),
    "precision_macro": float(macro_precision),
    "recall_macro": float(macro_recall),
    "f1_macro": float(macro_f1),
    "precision_weighted": float(weighted_precision),
    "recall_weighted": float(weighted_recall),
    "f1_weighted": float(weighted_f1),
    "true_distribution": {
        class_names[i]: int(true_counts[i])
        for i in range(len(class_names))
    },
    "predicted_distribution": {
        class_names[i]: int(pred_counts[i])
        for i in range(len(class_names))
    },
    "per_class": {
        name: {
            "precision": float(
                report[name]["precision"]
            ),
            "recall": float(
                report[name]["recall"]
            ),
            "f1": float(
                report[name]["f1-score"]
            ),
            "support": int(
                report[name]["support"]
            ),
        }
        for name in class_names
    },
    "synthetic_data": False,
    "model_retrained": False,
}

metrics_path = (
    DIAG_DIR
    / "validation_metrics_best_checkpoint.json"
)

with open(
    metrics_path,
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
    index=[
        f"Actual_{name}"
        for name in class_names
    ],
    columns=[
        f"Predicted_{name}"
        for name in class_names
    ],
)

cm_path = (
    DIAG_DIR
    / "validation_confusion_matrix_best_checkpoint.csv"
)

cm_df.to_csv(
    cm_path,
    index=True,
)

distribution_df = pd.DataFrame({
    "class": class_names,
    "true_count": [
        int(x)
        for x in true_counts
    ],
    "predicted_count": [
        int(x)
        for x in pred_counts
    ],
})

distribution_path = (
    DIAG_DIR
    / "validation_class_distribution_best_checkpoint.csv"
)

distribution_df.to_csv(
    distribution_path,
    index=False,
)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print(f"Metrics: {metrics_path}")
print(f"Confusion matrix: {cm_path}")
print(f"Class distribution: {distribution_path}")

print("\nINTEGRITY CHECKS")
print(f"Confusion matrix rows: {cm_total:,} — PASS")
print(f"True distribution rows: {true_total:,} — PASS")
print(f"Predicted rows: {pred_total:,} — PASS")

print("\nNo training was performed.")
print("Original model was not modified.")
print("Diagnostic artifacts were saved separately.")
