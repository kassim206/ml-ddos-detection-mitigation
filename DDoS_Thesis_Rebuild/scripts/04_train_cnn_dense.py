from pathlib import Path
import json
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BASE_DIR / "runs"
EXP001_DIR = RUNS_DIR / "experiment_001_dataset_char"
EXP002_DIR = RUNS_DIR / "experiment_002_preprocessing"
EXP003_DIR = RUNS_DIR / "experiment_003_baselines"
OUTPUT_DIR = RUNS_DIR / "experiment_004_cnn_dense"

TRAIN_FILE = EXP002_DIR / "train_processed.parquet"
VAL_FILE = EXP002_DIR / "val_processed.parquet"
TEST_FILE = EXP002_DIR / "test_processed.parquet"
SCHEMA_FILE = EXP001_DIR / "schema_discovery_report.json"
RF_METRICS_FILE = EXP003_DIR / "metrics_rf.json"
XGB_METRICS_FILE = EXP003_DIR / "metrics_xgb.json"
RF_CM_FILE = EXP003_DIR / "confusion_matrix_rf.csv"
XGB_CM_FILE = EXP003_DIR / "confusion_matrix_xgb.csv"

BEST_MODEL_FILE = OUTPUT_DIR / "best_model.keras"
HISTORY_FILE = OUTPUT_DIR / "training_history.json"
METRICS_FILE = OUTPUT_DIR / "metrics_cnn_lstm.json"
CONFUSION_FILE = OUTPUT_DIR / "confusion_matrix_cnn_lstm.csv"
CONFIG_FILE = OUTPUT_DIR / "training_config.json"
LABEL_MAP_FILE = OUTPUT_DIR / "label_mapping.json"

EXPECTED_TRAIN_ROWS = 7_702_608
EXPECTED_VAL_ROWS = 1_236_682
EXPECTED_TEST_ROWS = 1_773_902
EXPECTED_FEATURES = 39
EXPECTED_CLASSES = ["Benign", "DDoS", "Recon", "Web"]
LABEL_COLUMN = "Category"

BATCH_SIZE = 64
MAX_EPOCHS = 20
PATIENCE = 10
LEARNING_RATE = 1e-4
CLIPNORM = 1.0
MAX_CLASS_WEIGHT = 5.0
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


def fail(message):
    raise RuntimeError(message)


def load_json(path):
    if not path.exists() or path.stat().st_size == 0:
        fail(f"Required file not found or empty: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def validate_required_files():
    files = [TRAIN_FILE, VAL_FILE, TEST_FILE, SCHEMA_FILE, RF_METRICS_FILE, XGB_METRICS_FILE, RF_CM_FILE, XGB_CM_FILE]
    missing = [str(path) for path in files if not path.exists() or path.stat().st_size == 0]
    if missing:
        fail("Missing or empty required files:\n" + "\n".join(missing))


def discover_features():
    schema = load_json(SCHEMA_FILE)
    candidates = [schema.get("features"), schema.get("feature_columns"), schema.get("numeric_features"), schema.get("selected_features"), schema.get("columns")]
    valid = []
    for candidate in candidates:
        if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
            if len(candidate) == EXPECTED_FEATURES:
                valid.append(candidate)
    if not valid:
        fail(f"Unable to discover exactly {EXPECTED_FEATURES} features from {SCHEMA_FILE}")
    if len({tuple(x) for x in valid}) != 1:
        fail("Schema discovery fields disagree about the 39-feature order")
    features = valid[0]
    if LABEL_COLUMN in features:
        fail("Target column appears in the feature list")
    if len(set(features)) != EXPECTED_FEATURES:
        fail("Feature list contains duplicate columns")
    return features


def validate_dataset(path, features, expected_rows):
    parquet_file = pq.ParquetFile(path)
    columns = parquet_file.schema_arrow.names
    missing = [x for x in features + [LABEL_COLUMN] if x not in columns]
    if missing:
        fail(f"{path.name} missing required columns: {', '.join(missing)}")
    actual_rows = parquet_file.metadata.num_rows
    if actual_rows != expected_rows:
        fail(f"{path.name}: expected {expected_rows:,} rows, found {actual_rows:,}")
    counts = {x: 0 for x in EXPECTED_CLASSES}
    processed = 0
    for batch in parquet_file.iter_batches(batch_size=65536, columns=[LABEL_COLUMN], use_threads=False):
        labels = batch.column(0).to_pylist()
        for label in labels:
            label = str(label)
            if label not in counts:
                fail(f"{path.name}: unexpected class '{label}'")
            counts[label] += 1
        processed += len(labels)
    if processed != expected_rows:
        fail(f"{path.name}: processed {processed:,} rows instead of {expected_rows:,}")
    if any(counts[x] == 0 for x in EXPECTED_CLASSES):
        fail(f"{path.name}: at least one expected class is absent")
    return counts


def validate_baselines():
    rf = load_json(RF_METRICS_FILE)
    xgb = load_json(XGB_METRICS_FILE)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    for name, data in [("RF", rf), ("XGB", xgb)]:
        for metric in metrics:
            if metric not in data or not isinstance(data[metric], (int, float)):
                fail(f"{name} baseline metric '{metric}' is missing or non-numeric")
    for name, path in [("RF", RF_CM_FILE), ("XGB", XGB_CM_FILE)]:
        matrix = pd.read_csv(path, index_col=0).to_numpy()
        if matrix.shape != (4, 4):
            fail(f"{name} confusion matrix shape is {matrix.shape}, expected (4, 4)")
        if int(matrix.sum()) != EXPECTED_TEST_ROWS:
            fail(f"{name} confusion matrix total is {int(matrix.sum()):,}, expected {EXPECTED_TEST_ROWS:,}")
    return rf, xgb


def build_encoder():
    encoder = LabelEncoder()
    encoder.fit(EXPECTED_CLASSES)
    if list(encoder.classes_) != EXPECTED_CLASSES:
        fail(f"Unexpected class order: {encoder.classes_.tolist()}")
    return encoder, {name: i for i, name in enumerate(EXPECTED_CLASSES)}


def calculate_class_weights(train_counts):
    total = sum(train_counts.values())
    classes = len(EXPECTED_CLASSES)
    raw = {}
    applied = {}
    for index, name in enumerate(EXPECTED_CLASSES):
        weight = total / (classes * train_counts[name])
        raw[index] = float(weight)
        applied[index] = float(min(weight, MAX_CLASS_WEIGHT))
    return raw, applied


def parquet_generator(path, features, encoder):
    parquet_file = pq.ParquetFile(path)
    columns = features + [LABEL_COLUMN]
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=columns, use_threads=False):
        x = np.column_stack([batch.column(i).to_numpy(zero_copy_only=False) for i in range(len(features))]).astype(np.float32, copy=False)
        labels = np.asarray(batch.column(len(features)).to_pylist(), dtype=object)
        y = encoder.transform(labels.astype(str)).astype(np.int32, copy=False)
        yield x.reshape(-1, EXPECTED_FEATURES, 1), y


def create_dataset(path, features, encoder):
    signature = (
        tf.TensorSpec(shape=(None, EXPECTED_FEATURES, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
    return tf.data.Dataset.from_generator(lambda: parquet_generator(path, features, encoder), output_signature=signature).repeat().prefetch(1)


def create_model():
    inputs = tf.keras.Input(shape=(EXPECTED_FEATURES, 1), name="features")
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d")(inputs)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(len(EXPECTED_CLASSES), activation="softmax", name="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CNN_Dense")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def steps(rows):
    return (rows + BATCH_SIZE - 1) // BATCH_SIZE


def evaluate(model, dataset, test_steps):
    y_true_parts = []
    y_pred_parts = []
    processed = 0
    for x_batch, y_batch in dataset.take(test_steps):
        predictions = np.argmax(model(x_batch, training=False).numpy(), axis=1)
        truth = y_batch.numpy()
        y_true_parts.append(truth)
        y_pred_parts.append(predictions)
        processed += len(predictions)
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    if processed != EXPECTED_TEST_ROWS or len(y_true) != EXPECTED_TEST_ROWS or len(y_pred) != EXPECTED_TEST_ROWS:
        fail(f"Test evaluation processed {processed:,} rows; expected {EXPECTED_TEST_ROWS:,}")
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(EXPECTED_CLASSES)))
    if matrix.shape != (4, 4) or int(matrix.sum()) != EXPECTED_TEST_ROWS:
        fail("CNN confusion matrix validation failed")
    return y_pred, metrics, matrix


def compare(cnn, rf, xgb):
    result = {}
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        c = float(cnn[metric])
        r = float(rf[metric])
        x = float(xgb[metric])
        result[metric] = {"cnn_dense": c, "random_forest": r, "xgboost": x, "exceeds_random_forest": c > r, "exceeds_xgboost": c > x}
    passed = all(v["exceeds_random_forest"] and v["exceeds_xgboost"] for v in result.values())
    return result, passed


def backup_previous_outputs():
    existing = [BEST_MODEL_FILE, HISTORY_FILE, METRICS_FILE, CONFUSION_FILE, CONFIG_FILE, LABEL_MAP_FILE]
    if not any(path.exists() for path in existing):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = RUNS_DIR / f"experiment_004_cnn_dense_previous_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for path in existing:
        if path.exists():
            shutil.copy2(path, backup / path.name)
    return backup


def validate_outputs():
    required = [BEST_MODEL_FILE, HISTORY_FILE, METRICS_FILE, CONFUSION_FILE, CONFIG_FILE, LABEL_MAP_FILE]
    missing = [str(x) for x in required if not x.exists() or x.stat().st_size == 0]
    if missing:
        fail("Missing final Experiment 004 artifacts:\n" + "\n".join(missing))
    matrix = pd.read_csv(CONFUSION_FILE, index_col=0).to_numpy()
    if matrix.shape != (4, 4) or int(matrix.sum()) != EXPECTED_TEST_ROWS:
        fail("Saved CNN confusion matrix validation failed")
    model = tf.keras.models.load_model(BEST_MODEL_FILE)
    if model.input_shape != (None, EXPECTED_FEATURES, 1) or model.output_shape != (None, len(EXPECTED_CLASSES)):
        fail(f"Saved model shapes are {model.input_shape} -> {model.output_shape}")


def main():
    start = time.time()
    print("=" * 80)
    print("EXPERIMENT 004 — CNN-DENSE TRAINING — CORRECTED RERUN")
    print("=" * 80)
    validate_required_files()
    features = discover_features()
    print(f"Dynamically discovered features: {len(features)}")
    print(f"Target column: {LABEL_COLUMN}")
    print("Validating datasets...")
    train_counts = validate_dataset(TRAIN_FILE, features, EXPECTED_TRAIN_ROWS)
    val_counts = validate_dataset(VAL_FILE, features, EXPECTED_VAL_ROWS)
    test_counts = validate_dataset(TEST_FILE, features, EXPECTED_TEST_ROWS)
    rf_metrics, xgb_metrics = validate_baselines()
    print(f"Training rows: {EXPECTED_TRAIN_ROWS:,}")
    print(f"Validation rows: {EXPECTED_VAL_ROWS:,}")
    print(f"Test rows: {EXPECTED_TEST_ROWS:,}")
    print(f"Features: {len(features)}")
    for name, counts in [("Training", train_counts), ("Validation", val_counts), ("Test", test_counts)]:
        print(f"{name} class distribution:")
        for cls in EXPECTED_CLASSES:
            print(f"  {cls}: {counts[cls]:,}")
    print(f"RF   Accuracy={rf_metrics['accuracy']:.6f} Precision={rf_metrics['precision_macro']:.6f} Recall={rf_metrics['recall_macro']:.6f} F1={rf_metrics['f1_macro']:.6f}")
    print(f"XGB  Accuracy={xgb_metrics['accuracy']:.6f} Precision={xgb_metrics['precision_macro']:.6f} Recall={xgb_metrics['recall_macro']:.6f} F1={xgb_metrics['f1_macro']:.6f}")
    encoder, mapping = build_encoder()
    raw_weights, applied_weights = calculate_class_weights(train_counts)
    print("Raw inverse-frequency weights:")
    for i, cls in enumerate(EXPECTED_CLASSES):
        print(f"  {cls}: {raw_weights[i]:.6f}")
    print("Applied capped class weights:")
    for i, cls in enumerate(EXPECTED_CLASSES):
        print(f"  {cls}: {applied_weights[i]:.6f}")
    train_steps = steps(EXPECTED_TRAIN_ROWS)
    val_steps = steps(EXPECTED_VAL_ROWS)
    test_steps = steps(EXPECTED_TEST_ROWS)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training steps/epoch: {train_steps:,}")
    print(f"Validation steps/epoch: {val_steps:,}")
    print(f"Test steps: {test_steps:,}")
    print(f"Maximum epochs: {MAX_EPOCHS}")
    print(f"Early stopping patience: {PATIENCE}")
    backup = backup_previous_outputs()
    if backup:
        print(f"Previous Experiment 004 artifacts backed up to: {backup}")
    training_dataset = create_dataset(TRAIN_FILE, features, encoder)
    validation_dataset = create_dataset(VAL_FILE, features, encoder)
    test_dataset = create_dataset(TEST_FILE, features, encoder)
    model = create_model()
    model.summary()
    configuration = {
        "experiment": "004",
        "model": "CNN-Dense",
        "status": "completed_rerun_pending",
        "dataset_source": {"train": str(TRAIN_FILE), "validation": str(VAL_FILE), "test": str(TEST_FILE)},
        "target_column": LABEL_COLUMN,
        "features": features,
        "feature_count": EXPECTED_FEATURES,
        "classes": EXPECTED_CLASSES,
        "row_counts": {"train": EXPECTED_TRAIN_ROWS, "validation": EXPECTED_VAL_ROWS, "test": EXPECTED_TEST_ROWS},
        "architecture": {"input_shape": [EXPECTED_FEATURES, 1], "conv1d_filters": 64, "conv1d_kernel_size": 3, "conv1d_padding": "same", "conv1d_activation": "relu", "flatten": True, "dense_units": 128, "dense_activation": "relu", "dropout": 0.2, "output_units": 4, "output_activation": "softmax"},
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "clipnorm": CLIPNORM},
        "training": {"batch_size": BATCH_SIZE, "maximum_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE, "checkpoint_monitor": "val_accuracy", "checkpoint_mode": "max", "restore_best_weights": True, "streaming": True, "dataset_repeat": True, "random_seed": RANDOM_SEED},
        "class_weighting": {"method": "inverse_frequency_capped", "maximum_weight": MAX_CLASS_WEIGHT, "raw_weights": {str(k): v for k, v in raw_weights.items()}, "applied_weights": {str(k): v for k, v in applied_weights.items()}},
        "previous_artifacts_backup": str(backup) if backup else None,
    }
    save_json(CONFIG_FILE, configuration)
    save_json(LABEL_MAP_FILE, mapping)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(BEST_MODEL_FILE), monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=False, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=PATIENCE, restore_best_weights=True, verbose=1)
    print("Starting CNN-Dense training...")
    try:
        history = model.fit(training_dataset, steps_per_epoch=train_steps, validation_data=validation_dataset, validation_steps=val_steps, epochs=MAX_EPOCHS, class_weight=applied_weights, callbacks=[checkpoint, early_stopping], verbose=1)
    except KeyboardInterrupt:
        print("Training interrupted by user. No completed rerun metrics are being generated.")
        print(f"Best checkpoint retained at: {BEST_MODEL_FILE}")
        return
    if not BEST_MODEL_FILE.exists():
        fail("Best model checkpoint was not created")
    history_data = {k: [float(v) for v in values] for k, values in history.history.items()}
    save_json(HISTORY_FILE, history_data)
    best_model = tf.keras.models.load_model(BEST_MODEL_FILE)
    y_pred, cnn_metrics, matrix = evaluate(best_model, test_dataset, test_steps)
    comparison, passed = compare(cnn_metrics, rf_metrics, xgb_metrics)
    metrics_output = {"experiment": "004", "model": "CNN-Dense", "dataset_source": {"train": str(TRAIN_FILE), "validation": str(VAL_FILE), "test": str(TEST_FILE)}, "target_column": LABEL_COLUMN, "test_rows": EXPECTED_TEST_ROWS, "features": EXPECTED_FEATURES, "classes": EXPECTED_CLASSES, **cnn_metrics, "predictions": int(len(y_pred)), "confusion_matrix_total": int(matrix.sum()), "baseline_comparison": comparison, "performance_floor_passed": passed, "epochs_executed": len(history_data.get("loss", [])), "runtime_seconds": time.time() - start}
    save_json(METRICS_FILE, metrics_output)
    pd.DataFrame(matrix, index=EXPECTED_CLASSES, columns=EXPECTED_CLASSES).rename_axis("Actual").to_csv(CONFUSION_FILE)
    configuration["status"] = "SUCCESS"
    configuration["epochs_executed"] = len(history_data.get("loss", []))
    configuration["performance_floor_passed"] = passed
    save_json(CONFIG_FILE, configuration)
    validate_outputs()
    print("=" * 80)
    print("EXPERIMENT 004 RESULTS")
    print("=" * 80)
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        print(f"{key}: {cnn_metrics[key]:.6f}")
    print(f"Predictions: {len(y_pred):,}")
    print(f"Confusion matrix total: {int(matrix.sum()):,}")
    print(f"Epochs executed: {len(history_data.get('loss', []))}")
    print(f"Performance floor passed: {passed}")
    print("Artifact validation: PASSED")
    print(f"Runtime seconds: {time.time() - start:.2f}")

                
if __name__ == "__main__":
    main()
