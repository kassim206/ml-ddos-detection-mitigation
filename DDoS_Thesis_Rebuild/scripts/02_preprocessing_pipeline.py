import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

EXPERIMENT_DIR = Path("runs/experiment_001_dataset_char/cleaned_data")
OUTPUT_DIR = Path("runs/experiment_002_preprocessing")

SCHEMA_FILE = Path(
    "runs/experiment_001_dataset_char/schema_discovery_report.json"
)

RANDOM_SEED = 42
CHUNK_SIZE = 100_000
MERGE_BATCH_SIZE = 500_000

EXPECTED_ROWS = {
    "train": 7_702_608,
    "val": 1_236_682,
    "test": 1_773_902,
}

EXPECTED_TOTAL = sum(EXPECTED_ROWS.values())

np.random.seed(RANDOM_SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not EXPERIMENT_DIR.exists():
    raise FileNotFoundError(
        f"Experiment 001 cleaned data directory not found: {EXPERIMENT_DIR}"
    )

if not SCHEMA_FILE.exists():
    raise FileNotFoundError(
        f"Schema discovery report not found: {SCHEMA_FILE}"
    )

with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
    schema = json.load(f)

X_COLUMNS = schema["features"]

if not X_COLUMNS:
    raise ValueError("No features found in schema_discovery_report.json")

CATEGORY_COLUMN = "Category"
ALL_OUTPUT_COLUMNS = X_COLUMNS + [CATEGORY_COLUMN]


def get_csv_files(split_name):
    split_dir = EXPERIMENT_DIR / split_name

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}"
        )

    files = sorted(split_dir.glob("*.csv"))

    if not files:
        raise ValueError(
            f"No CSV files found in {split_dir}"
        )

    return files


def count_rows(split_name):
    total = 0

    for file_path in get_csv_files(split_name):
        for chunk in pd.read_csv(
            file_path,
            usecols=lambda c: c in ALL_OUTPUT_COLUMNS,
            chunksize=CHUNK_SIZE
        ):
            total += len(chunk)

    return total


def validate_input_structure(split_name):
    files = get_csv_files(split_name)

    for file_path in files:
        header = pd.read_csv(file_path, nrows=0)

        missing_features = [
            col for col in X_COLUMNS
            if col not in header.columns
        ]

        if missing_features:
            raise ValueError(
                f"{file_path} is missing required features: "
                f"{missing_features}"
            )

        if CATEGORY_COLUMN not in header.columns:
            raise ValueError(
                f"{file_path} does not contain '{CATEGORY_COLUMN}'"
            )


def fit_scaler():
    scaler = StandardScaler()
    training_rows = 0

    print("[1/4] Fitting StandardScaler on training data...")

    for file_path in get_csv_files("train"):
        print(f"    {file_path.name}")

        for chunk in pd.read_csv(
            file_path,
            usecols=X_COLUMNS,
            chunksize=CHUNK_SIZE
        ):
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk[X_COLUMNS] = chunk[X_COLUMNS].fillna(0)

            scaler.partial_fit(chunk[X_COLUMNS])
            training_rows += len(chunk)

    if training_rows != EXPECTED_ROWS["train"]:
        raise RuntimeError(
            f"Training scaler row count mismatch. "
            f"Expected {EXPECTED_ROWS['train']:,}, "
            f"found {training_rows:,}"
        )

    scaler_path = OUTPUT_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    print(
        f"    Training rows used for scaler: "
        f"{training_rows:,}"
    )

    return scaler


def process_split(split_name, scaler):
    expected_rows = EXPECTED_ROWS[split_name]

    output_path = OUTPUT_DIR / f"{split_name}_processed.parquet"
    temp_dir = OUTPUT_DIR / f"_temp_{split_name}"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    print(
        f"[2/4] Processing {split_name}: "
        f"expected {expected_rows:,} rows"
    )

    written_rows = 0
    part_index = 0

    for file_path in get_csv_files(split_name):
        print(f"    Processing {file_path.name}")

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE
        ):
            missing_features = [
                col for col in X_COLUMNS
                if col not in chunk.columns
            ]

            if missing_features:
                raise ValueError(
                    f"{file_path} is missing features: "
                    f"{missing_features}"
                )

            if CATEGORY_COLUMN not in chunk.columns:
                raise ValueError(
                    f"{file_path} is missing Category"
                )

            original_chunk_rows = len(chunk)

            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk[X_COLUMNS] = chunk[X_COLUMNS].fillna(0)

            transformed = scaler.transform(chunk[X_COLUMNS])

            transformed_df = pd.DataFrame(
                transformed,
                columns=X_COLUMNS,
                index=chunk.index
            )

            transformed_df[CATEGORY_COLUMN] = (
                chunk[CATEGORY_COLUMN].values
            )

            transformed_df = transformed_df[
                ALL_OUTPUT_COLUMNS
            ]

            if len(transformed_df) != original_chunk_rows:
                raise RuntimeError(
                    f"Row loss detected in {file_path}. "
                    f"Input chunk: {original_chunk_rows:,}, "
                    f"output chunk: {len(transformed_df):,}"
                )

            temp_file = (
                temp_dir /
                f"part_{part_index:06d}.parquet"
            )

            transformed_df.to_parquet(
                temp_file,
                index=False
            )

            written_rows += len(transformed_df)
            part_index += 1

    if written_rows != expected_rows:
        raise RuntimeError(
            f"{split_name} row count mismatch after processing. "
            f"Expected {expected_rows:,}, "
            f"found {written_rows:,}"
        )

    temp_files = sorted(temp_dir.glob("part_*.parquet"))

    if not temp_files:
        raise RuntimeError(
            f"No temporary Parquet files created for {split_name}"
        )

    print(
        f"    Temporary files created: {len(temp_files)}"
    )

    merge_parquet_files(
        temp_files,
        output_path,
        expected_rows,
        split_name
    )

    if output_path.exists():
        final_rows = count_parquet_rows(output_path)

        if final_rows != expected_rows:
            raise RuntimeError(
                f"{split_name} final Parquet row count mismatch. "
                f"Expected {expected_rows:,}, "
                f"found {final_rows:,}"
            )

    shutil.rmtree(temp_dir)

    print(
        f"    {split_name.capitalize()} complete: "
        f"{written_rows:,} rows"
    )

    return [written_rows, len(ALL_OUTPUT_COLUMNS)]


def merge_parquet_files(
    temp_files,
    output_path,
    expected_rows,
    split_name
):
    print(
        f"    Merging {len(temp_files)} temporary files..."
    )

    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    merged_rows = 0
    batch = []
    batch_rows = 0

    try:
        for temp_file in temp_files:
            df = pd.read_parquet(temp_file)

            batch.append(df)
            batch_rows += len(df)

            if batch_rows >= MERGE_BATCH_SIZE:
                combined = pd.concat(
                    batch,
                    ignore_index=True
                )

                table = pa.Table.from_pandas(
                    combined,
                    preserve_index=False
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema
                    )

                writer.write_table(table)

                merged_rows += len(combined)

                del combined
                del table
                batch = []
                batch_rows = 0

        if batch:
            combined = pd.concat(
                batch,
                ignore_index=True
            )

            table = pa.Table.from_pandas(
                combined,
                preserve_index=False
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema
                )

            writer.write_table(table)

            merged_rows += len(combined)

            del combined
            del table

    finally:
        if writer is not None:
            writer.close()

    if merged_rows != expected_rows:
        if output_path.exists():
            output_path.unlink()

        raise RuntimeError(
            f"{split_name} merge row count mismatch. "
            f"Expected {expected_rows:,}, "
            f"found {merged_rows:,}"
        )

    print(
        f"    Merged rows: {merged_rows:,}"
    )


def count_parquet_rows(path):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)

    return parquet_file.metadata.num_rows


def verify_final_outputs():
    print("[4/4] Verifying final outputs...")

    results = {}
    total = 0

    for split_name, expected in EXPECTED_ROWS.items():
        path = OUTPUT_DIR / f"{split_name}_processed.parquet"

        if not path.exists():
            raise RuntimeError(
                f"Missing final output: {path}"
            )

        rows = count_parquet_rows(path)

        if rows != expected:
            raise RuntimeError(
                f"{split_name} verification failed. "
                f"Expected {expected:,}, "
                f"found {rows:,}"
            )

        results[split_name] = rows
        total += rows

        print(
            f"    {split_name}: "
            f"{rows:,} / {expected:,}"
        )

    if total != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Total row verification failed. "
            f"Expected {EXPECTED_TOTAL:,}, "
            f"found {total:,}"
        )

    print(
        f"    TOTAL: {total:,} / {EXPECTED_TOTAL:,}"
    )

    return results


def main():
    print("=" * 70)
    print("EXPERIMENT 002 - PREPROCESSING PIPELINE")
    print("=" * 70)

    print()
    print("Expected dataset:")
    print(f"    Train:      {EXPECTED_ROWS['train']:,}")
    print(f"    Validation: {EXPECTED_ROWS['val']:,}")
    print(f"    Test:       {EXPECTED_ROWS['test']:,}")
    print(f"    Total:      {EXPECTED_TOTAL:,}")
    print()

    print("[0/4] Validating input files...")

    for split_name in EXPECTED_ROWS:
        validate_input_structure(split_name)

        actual_rows = count_rows(split_name)

        print(
            f"    {split_name}: "
            f"{actual_rows:,} rows"
        )

        if actual_rows != EXPECTED_ROWS[split_name]:
            raise RuntimeError(
                f"{split_name} input row count mismatch. "
                f"Expected {EXPECTED_ROWS[split_name]:,}, "
                f"found {actual_rows:,}"
            )

    scaler = fit_scaler()

    train_shape = process_split("train", scaler)
    val_shape = process_split("val", scaler)
    test_shape = process_split("test", scaler)

    final_counts = verify_final_outputs()

    temp_dirs = list(
        OUTPUT_DIR.glob("_temp_*")
    )

    if temp_dirs:
        raise RuntimeError(
            f"Temporary directories remain: "
            f"{[str(x) for x in temp_dirs]}"
        )

    preprocessing_log = {
        "experiment_id": "002",
        "random_seed": RANDOM_SEED,
        "source_experiment": "001",
        "source_directory": str(EXPERIMENT_DIR),
        "synthetic_data": False,
        "mock_data": False,
        "sampling": False,
        "row_generation": False,
        "scaling_method": "StandardScaler partial_fit",
        "scaler_fit_source": "training_split_only",
        "output_format": "Parquet",
        "chunk_size": CHUNK_SIZE,
        "merge_batch_size": MERGE_BATCH_SIZE,
        "feature_count": len(X_COLUMNS),
        "output_column_count": len(ALL_OUTPUT_COLUMNS),
        "features_used": X_COLUMNS,
        "label_column": CATEGORY_COLUMN,
        "nan_handling": "NaN and infinite feature values replaced with 0 before scaling",
        "train_shape": train_shape,
        "val_shape": val_shape,
        "test_shape": test_shape,
        "train_rows": final_counts["train"],
        "validation_rows": final_counts["val"],
        "test_rows": final_counts["test"],
        "total_rows": sum(final_counts.values()),
        "expected_total_rows": EXPECTED_TOTAL,
        "row_count_verified": True,
        "data_loss_detected": False,
        "temporary_append_mode": False,
        "final_verification": "passed"
    }

    with open(
        OUTPUT_DIR / "preprocessing_log.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            preprocessing_log,
            f,
            indent=4
        )

    with open(
        OUTPUT_DIR / "config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "experiment_id": "002",
                "chunk_size": CHUNK_SIZE,
                "merge_batch_size": MERGE_BATCH_SIZE,
                "random_seed": RANDOM_SEED,
                "expected_rows": EXPECTED_ROWS,
                "expected_total_rows": EXPECTED_TOTAL,
                "feature_count": len(X_COLUMNS),
                "output_columns": ALL_OUTPUT_COLUMNS,
                "synthetic_data": False,
                "mock_data": False,
                "sampling": False
            },
            f,
            indent=4
        )

    print()
    print("=" * 70)
    print("[SUCCESS] Experiment 002 Complete")
    print("=" * 70)
    print(
        f"Train:      {final_counts['train']:,} rows"
    )
    print(
        f"Validation: {final_counts['val']:,} rows"
    )
    print(
        f"Test:       {final_counts['test']:,} rows"
    )
    print(
        f"TOTAL:      {sum(final_counts.values()):,} rows"
    )
    print(
        f"Expected:   {EXPECTED_TOTAL:,} rows"
    )
    print("Data loss:  0 rows")
    print("=" * 70)


if __name__ == "__main__":
    main()