import gc
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import tensorflow as tf

EXPERIMENT_ID = "006"

DATA_DIR = Path("runs/experiment_002_preprocessing")
EXP001_DIR = Path("runs/experiment_001_dataset_char")
EXP003_DIR = Path("runs/experiment_003_baselines")
EXP004_DIR = Path("runs/experiment_004_cnn_dense")
EXP009_DIR = Path("runs/experiment_009_hybrid_cnn_lstm")
OUTPUT_DIR = Path("runs/experiment_006_runtime")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = DATA_DIR / "test_processed.parquet"
OUTPUT_CSV = OUTPUT_DIR / "resource_usage.csv"
REPEATS_CSV = OUTPUT_DIR / "runtime_repeats.csv"
CONFIG_PATH = OUTPUT_DIR / "config.json"

EXPECTED_TEST_ROWS = 1773902
CHUNK_SIZE = 32768
REPEATS = 3
WARMUP_SIZE = 512
KERAS_BATCH_SIZE = 512

with open(
    EXP001_DIR / "schema_discovery_report.json",
    "r",
    encoding="utf-8"
) as f:
    X_COLUMNS = json.load(f)["features"]

if len(X_COLUMNS) != 39:
    raise ValueError(
        f"Expected 39 features, found {len(X_COLUMNS)}"
    )

if len(set(X_COLUMNS)) != 39:
    raise ValueError(
        "Duplicate feature names detected"
    )

if not TEST_PATH.exists():
    raise FileNotFoundError(TEST_PATH)

parquet_file = pq.ParquetFile(TEST_PATH)
test_rows = parquet_file.metadata.num_rows

if test_rows != EXPECTED_TEST_ROWS:
    raise ValueError(
        f"Unexpected test row count: "
        f"{test_rows} != {EXPECTED_TEST_ROWS}"
    )

parquet_columns = set(
    parquet_file.schema_arrow.names
)

missing_features = [
    feature
    for feature in X_COLUMNS
    if feature not in parquet_columns
]

if missing_features:
    raise ValueError(
        f"Missing test features: {missing_features}"
    )

rf_path = EXP003_DIR / "model_rf.pkl"
xgb_path = EXP003_DIR / "model_xgb.pkl"
cnn_path = EXP004_DIR / "best_model.keras"
hybrid_path = EXP009_DIR / "best_model.keras"

required_models = [
    rf_path,
    xgb_path,
    cnn_path,
    hybrid_path
]

missing_models = [
    str(path)
    for path in required_models
    if not path.exists()
]

if missing_models:
    raise FileNotFoundError(
        "Required model artifacts are missing:\n"
        + "\n".join(missing_models)
    )

process = psutil.Process()

results = []
repeat_results = []


def rss_mb():
    return (
        process.memory_info().rss
        / (1024 * 1024)
    )


def load_warmup_data():
    pf = pq.ParquetFile(TEST_PATH)

    for batch in pf.iter_batches(
        batch_size=WARMUP_SIZE,
        columns=X_COLUMNS
    ):
        df = batch.to_pandas()

        x = df.to_numpy(
            dtype=np.float32,
            copy=True
        )

        if len(x) > WARMUP_SIZE:
            x = x[:WARMUP_SIZE]

        if x.shape != (
            WARMUP_SIZE,
            len(X_COLUMNS)
        ):
            raise ValueError(
                f"Unexpected warm-up shape: "
                f"{x.shape}"
            )

        if not np.isfinite(x).all():
            raise ValueError(
                "Non-finite warm-up features detected"
            )

        return x

    raise RuntimeError(
        "Unable to read warm-up observations"
    )


WARMUP_X = load_warmup_data()


def benchmark_sklearn(
    name,
    model
):
    print()
    print(
        f"[{name}] Starting benchmark"
    )
    print(
        f"[{name}] Test rows: "
        f"{EXPECTED_TEST_ROWS:,}"
    )
    print(
        f"[{name}] Repeats: {REPEATS}"
    )

    print(
        f"[{name}] Warm-up..."
    )

    warmup_predictions = model.predict(
        WARMUP_X
    )

    if len(warmup_predictions) != len(
        WARMUP_X
    ):
        raise RuntimeError(
            f"{name} warm-up prediction "
            f"count mismatch"
        )

    del warmup_predictions
    gc.collect()

    timings = []
    peak_rss_values = []
    rss_delta_values = []

    for repeat in range(
        1,
        REPEATS + 1
    ):
        gc.collect()

        start_rss = rss_mb()
        peak_rss = start_rss
        processed = 0

        pf = pq.ParquetFile(
            TEST_PATH
        )

        print()
        print(
            f"[{name}] "
            f"Repeat {repeat}/{REPEATS}"
        )

        start_time = time.perf_counter()

        for batch in pf.iter_batches(
            batch_size=CHUNK_SIZE,
            columns=X_COLUMNS
        ):
            df = batch.to_pandas()

            x_chunk = df.to_numpy(
                dtype=np.float32,
                copy=True
            )

            del df

            if (
                x_chunk.ndim != 2
                or x_chunk.shape[1]
                != len(X_COLUMNS)
            ):
                raise ValueError(
                    f"{name}: unexpected "
                    f"chunk shape "
                    f"{x_chunk.shape}"
                )

            if not np.isfinite(
                x_chunk
            ).all():
                raise ValueError(
                    f"{name}: non-finite "
                    f"features detected"
                )

            predictions = model.predict(
                x_chunk
            )

            if len(predictions) != len(
                x_chunk
            ):
                raise RuntimeError(
                    f"{name}: prediction "
                    f"count mismatch"
                )

            processed += len(
                x_chunk
            )

            current_rss = rss_mb()

            if current_rss > peak_rss:
                peak_rss = current_rss

            print(
                f"\r[{name}] "
                f"Repeat {repeat}/{REPEATS}: "
                f"{processed:,}/"
                f"{EXPECTED_TEST_ROWS:,}",
                end="",
                flush=True
            )

            del predictions
            del x_chunk

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()

        if processed != EXPECTED_TEST_ROWS:
            raise RuntimeError(
                f"{name}: processed "
                f"{processed} rows; expected "
                f"{EXPECTED_TEST_ROWS}"
            )

        end_rss = rss_mb()

        rss_delta = max(
            0.0,
            peak_rss - start_rss
        )

        timings.append(
            elapsed
        )

        peak_rss_values.append(
            peak_rss
        )

        rss_delta_values.append(
            rss_delta
        )

        repeat_results.append(
            {
                "model": name,
                "repeat": repeat,
                "rows_processed": processed,
                "total_seconds": float(
                    elapsed
                ),
                "amortized_ms_per_sample": float(
                    elapsed
                    * 1000.0
                    / processed
                ),
                "throughput_samples_per_second": float(
                    processed
                    / elapsed
                ),
                "start_rss_mb": float(
                    start_rss
                ),
                "peak_rss_mb": float(
                    peak_rss
                ),
                "end_rss_mb": float(
                    end_rss
                ),
                "peak_rss_increase_mb": float(
                    rss_delta
                )
            }
        )

        print(
            f"[{name}] "
            f"Repeat {repeat} complete"
        )

        print(
            f"[{name}] "
            f"Elapsed: "
            f"{elapsed:.6f} seconds"
        )

        print(
            f"[{name}] "
            f"Throughput: "
            f"{processed / elapsed:,.2f} "
            f"samples/second"
        )

        print(
            f"[{name}] "
            f"Peak RSS increase: "
            f"{rss_delta:.2f} MB"
        )

        gc.collect()

    timings = np.asarray(
        timings,
        dtype=np.float64
    )

    peak_rss_values = np.asarray(
        peak_rss_values,
        dtype=np.float64
    )

    rss_delta_values = np.asarray(
        rss_delta_values,
        dtype=np.float64
    )

    mean_seconds = float(
        np.mean(timings)
    )

    result = {
        "model": name,
        "rows_per_repeat": int(
            EXPECTED_TEST_ROWS
        ),
        "repeats": int(
            REPEATS
        ),
        "total_rows_predicted_measured": int(
            EXPECTED_TEST_ROWS
            * REPEATS
        ),
        "mean_total_seconds": mean_seconds,
        "std_total_seconds": float(
            np.std(timings)
        ),
        "min_total_seconds": float(
            np.min(timings)
        ),
        "max_total_seconds": float(
            np.max(timings)
        ),
        "mean_amortized_ms_per_sample": float(
            mean_seconds
            * 1000.0
            / EXPECTED_TEST_ROWS
        ),
        "median_amortized_ms_per_sample": float(
            np.median(timings)
            * 1000.0
            / EXPECTED_TEST_ROWS
        ),
        "throughput_samples_per_second": float(
            EXPECTED_TEST_ROWS
            / mean_seconds
        ),
        "mean_peak_rss_mb": float(
            np.mean(
                peak_rss_values
            )
        ),
        "mean_peak_rss_increase_mb": float(
            np.mean(
                rss_delta_values
            )
        )
    }

    print()
    print(
        f"[{name}] Benchmark complete"
    )

    return result


def benchmark_keras(
    name,
    model
):
    expected_shape = tuple(
        model.input_shape
    )

    required_shape = (
        None,
        len(X_COLUMNS),
        1
    )

    if expected_shape != required_shape:
        raise ValueError(
            f"{name} expects "
            f"{expected_shape}; expected "
            f"{required_shape}"
        )

    print()
    print(
        f"[{name}] Starting benchmark"
    )
    print(
        f"[{name}] Test rows: "
        f"{EXPECTED_TEST_ROWS:,}"
    )
    print(
        f"[{name}] Repeats: {REPEATS}"
    )

    warmup_x = WARMUP_X.reshape(
        -1,
        len(X_COLUMNS),
        1
    )

    print(
        f"[{name}] Warm-up..."
    )

    warmup_predictions = model.predict(
        warmup_x,
        batch_size=KERAS_BATCH_SIZE,
        verbose=0
    )

    if len(
        warmup_predictions
    ) != len(warmup_x):
        raise RuntimeError(
            f"{name} warm-up prediction "
            f"count mismatch"
        )

    del warmup_predictions
    del warmup_x
    gc.collect()

    timings = []
    peak_rss_values = []
    rss_delta_values = []

    for repeat in range(
        1,
        REPEATS + 1
    ):
        gc.collect()

        start_rss = rss_mb()
        peak_rss = start_rss
        processed = 0

        pf = pq.ParquetFile(
            TEST_PATH
        )

        print()
        print(
            f"[{name}] "
            f"Repeat {repeat}/{REPEATS}"
        )

        start_time = time.perf_counter()

        for batch in pf.iter_batches(
            batch_size=CHUNK_SIZE,
            columns=X_COLUMNS
        ):
            df = batch.to_pandas()

            x_chunk = df.to_numpy(
                dtype=np.float32,
                copy=True
            )

            del df

            if (
                x_chunk.ndim != 2
                or x_chunk.shape[1]
                != len(X_COLUMNS)
            ):
                raise ValueError(
                    f"{name}: unexpected "
                    f"chunk shape "
                    f"{x_chunk.shape}"
                )

            if not np.isfinite(
                x_chunk
            ).all():
                raise ValueError(
                    f"{name}: non-finite "
                    f"features detected"
                )

            x_model = x_chunk.reshape(
                -1,
                len(X_COLUMNS),
                1
            )

            predictions = model.predict(
                x_model,
                batch_size=KERAS_BATCH_SIZE,
                verbose=0
            )

            if len(predictions) != len(
                x_model
            ):
                raise RuntimeError(
                    f"{name}: prediction "
                    f"count mismatch"
                )

            processed += len(
                x_model
            )

            current_rss = rss_mb()

            if current_rss > peak_rss:
                peak_rss = current_rss

            print(
                f"\r[{name}] "
                f"Repeat {repeat}/{REPEATS}: "
                f"{processed:,}/"
                f"{EXPECTED_TEST_ROWS:,}",
                end="",
                flush=True
            )

            del predictions
            del x_model
            del x_chunk

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()

        if processed != EXPECTED_TEST_ROWS:
            raise RuntimeError(
                f"{name}: processed "
                f"{processed} rows; expected "
                f"{EXPECTED_TEST_ROWS}"
            )

        end_rss = rss_mb()

        rss_delta = max(
            0.0,
            peak_rss - start_rss
        )

        timings.append(
            elapsed
        )

        peak_rss_values.append(
            peak_rss
        )

        rss_delta_values.append(
            rss_delta
        )

        repeat_results.append(
            {
                "model": name,
                "repeat": repeat,
                "rows_processed": processed,
                "total_seconds": float(
                    elapsed
                ),
                "amortized_ms_per_sample": float(
                    elapsed
                    * 1000.0
                    / processed
                ),
                "throughput_samples_per_second": float(
                    processed
                    / elapsed
                ),
                "start_rss_mb": float(
                    start_rss
                ),
                "peak_rss_mb": float(
                    peak_rss
                ),
                "end_rss_mb": float(
                    end_rss
                ),
                "peak_rss_increase_mb": float(
                    rss_delta
                )
            }
        )

        print(
            f"[{name}] "
            f"Repeat {repeat} complete"
        )

        print(
            f"[{name}] "
            f"Elapsed: "
            f"{elapsed:.6f} seconds"
        )

        print(
            f"[{name}] "
            f"Throughput: "
            f"{processed / elapsed:,.2f} "
            f"samples/second"
        )

        print(
            f"[{name}] "
            f"Peak RSS increase: "
            f"{rss_delta:.2f} MB"
        )

        gc.collect()

    timings = np.asarray(
        timings,
        dtype=np.float64
    )

    peak_rss_values = np.asarray(
        peak_rss_values,
        dtype=np.float64
    )

    rss_delta_values = np.asarray(
        rss_delta_values,
        dtype=np.float64
    )

    mean_seconds = float(
        np.mean(timings)
    )

    result = {
        "model": name,
        "rows_per_repeat": int(
            EXPECTED_TEST_ROWS
        ),
        "repeats": int(
            REPEATS
        ),
        "total_rows_predicted_measured": int(
            EXPECTED_TEST_ROWS
            * REPEATS
        ),
        "mean_total_seconds": mean_seconds,
        "std_total_seconds": float(
            np.std(timings)
        ),
        "min_total_seconds": float(
            np.min(timings)
        ),
        "max_total_seconds": float(
            np.max(timings)
        ),
        "mean_amortized_ms_per_sample": float(
            mean_seconds
            * 1000.0
            / EXPECTED_TEST_ROWS
        ),
        "median_amortized_ms_per_sample": float(
            np.median(timings)
            * 1000.0
            / EXPECTED_TEST_ROWS
        ),
        "throughput_samples_per_second": float(
            EXPECTED_TEST_ROWS
            / mean_seconds
        ),
        "mean_peak_rss_mb": float(
            np.mean(
                peak_rss_values
            )
        ),
        "mean_peak_rss_increase_mb": float(
            np.mean(
                rss_delta_values
            )
        )
    }

    print()
    print(
        f"[{name}] Benchmark complete"
    )

    return result


print()
print(
    "EXPERIMENT 006 — FULL TEST-SET "
    "RUNTIME BENCHMARK"
)
print()

print(
    f"Test source: {TEST_PATH}"
)
print(
    f"Verified test rows: "
    f"{test_rows:,}"
)
print(
    f"Features: {len(X_COLUMNS)}"
)
print(
    f"Chunk size: {CHUNK_SIZE:,}"
)
print(
    f"Measured repeats: {REPEATS}"
)
print(
    "Sampling: False"
)
print()

print(
    "[1/4] Loading Random Forest..."
)

rf_model = joblib.load(
    rf_path
)

results.append(
    benchmark_sklearn(
        "RandomForest",
        rf_model
    )
)

del rf_model
gc.collect()

print()
print(
    "[1/4] Random Forest finished"
)

print()
print(
    "[2/4] Loading XGBoost..."
)

xgb_model = joblib.load(
    xgb_path
)

results.append(
    benchmark_sklearn(
        "XGBoost",
        xgb_model
    )
)

del xgb_model
gc.collect()

print()
print(
    "[2/4] XGBoost finished"
)

print()
print(
    "[3/4] Loading CNN-Dense..."
)

tf.keras.backend.clear_session()
gc.collect()

cnn_model = tf.keras.models.load_model(
    cnn_path
)

results.append(
    benchmark_keras(
        "CNN-Dense",
        cnn_model
    )
)

del cnn_model
tf.keras.backend.clear_session()
gc.collect()

print()
print(
    "[3/4] CNN-Dense finished"
)

print()
print(
    "[4/4] Loading Hybrid CNN-LSTM..."
)

hybrid_model = tf.keras.models.load_model(
    hybrid_path
)

results.append(
    benchmark_keras(
        "Hybrid-CNN-LSTM",
        hybrid_model
    )
)

del hybrid_model
tf.keras.backend.clear_session()
gc.collect()

print()
print(
    "[4/4] Hybrid CNN-LSTM finished"
)

results_df = pd.DataFrame(
    results
)

repeat_results_df = pd.DataFrame(
    repeat_results
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

repeat_results_df.to_csv(
    REPEATS_CSV,
    index=False
)

config = {
    "experiment_id": EXPERIMENT_ID,
    "status": "completed",
    "benchmark_scope": (
        "complete_processed_test_split"
    ),
    "benchmark_type": (
        "chunked_batched_cpu_inference"
    ),
    "test_data_source": str(
        TEST_PATH
    ),
    "test_rows": int(
        test_rows
    ),
    "expected_test_rows": int(
        EXPECTED_TEST_ROWS
    ),
    "feature_count": int(
        len(X_COLUMNS)
    ),
    "sampling": False,
    "all_test_rows_processed": True,
    "chunked_processing": True,
    "chunk_size": int(
        CHUNK_SIZE
    ),
    "measured_repeats": int(
        REPEATS
    ),
    "warmup_samples": int(
        WARMUP_SIZE
    ),
    "keras_batch_size": int(
        KERAS_BATCH_SIZE
    ),
    "rows_predicted_per_model_measured": int(
        EXPECTED_TEST_ROWS
        * REPEATS
    ),
    "synthetic_latency": False,
    "synthetic_resource_usage": False,
    "hardcoded_runtime_results": False,
    "latency_interpretation": (
        "Amortized inference time per "
        "observation derived from the elapsed "
        "time required to process the complete "
        "test split. It is not independent "
        "single-request latency."
    ),
    "throughput_interpretation": (
        "Number of complete test observations "
        "processed per second during chunked "
        "batched inference."
    ),
    "memory_interpretation": (
        "Process RSS measurements observed "
        "during inference. Peak RSS increase "
        "is relative to process RSS at the "
        "start of each measured repeat and is "
        "not total model memory requirement."
    ),
    "platform": platform.platform(),
    "python_process_pid": int(
        process.pid
    ),
    "tensorflow_version": tf.__version__,
    "models": {
        "RandomForest": str(
            rf_path
        ),
        "XGBoost": str(
            xgb_path
        ),
        "CNN-Dense": str(
            cnn_path
        ),
        "Hybrid-CNN-LSTM": str(
            hybrid_path
        )
    }
}

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=2
    )

print()
print(
    "EXPERIMENT 006 RESULTS"
)
print()

print(
    results_df.to_string(
        index=False
    )
)

print()
print(
    f"Saved summary: {OUTPUT_CSV}"
)
print(
    f"Saved individual repeats: "
    f"{REPEATS_CSV}"
)
print(
    f"Saved configuration: "
    f"{CONFIG_PATH}"
)
print()
print(
    "Experiment 006 completed successfully."
)