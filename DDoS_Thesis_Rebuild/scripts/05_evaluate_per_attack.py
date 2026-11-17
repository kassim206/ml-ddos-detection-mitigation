import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXPERIMENT_ID = "005"

OUTPUT_DIR = Path("runs/experiment_005_per_attack")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXP001_DIR = Path("runs/experiment_001_dataset_char")
EXP009_DIR = Path("runs/experiment_009_hybrid_cnn_lstm")

PREDICTIONS_PATH = EXP009_DIR / "test_predictions.parquet"
TEST_PROVENANCE_PATH = (
    EXP001_DIR
    / "cleaned_data"
    / "test"
    / "test_combined.csv"
)

OUTPUT_JSON = OUTPUT_DIR / "attack_family_metrics.json"
OUTPUT_CSV = OUTPUT_DIR / "attack_family_metrics.csv"
CONFIG_PATH = OUTPUT_DIR / "config.json"

EXPECTED_TEST_ROWS = 1773902

EXPECTED_ATTACKS = [
    "Benign_Final",
    "DDoS-HTTP_Flood",
    "DDoS-SlowLoris",
    "DDoS-SYN_Flood",
    "DDoS-UDP_Flood",
    "Recon-PortScan",
    "SqlInjection",
    "XSS"
]

CATEGORY_MAP = {
    "Benign_Final": "Benign",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "Recon-PortScan": "Recon",
    "SqlInjection": "Web",
    "XSS": "Web"
}

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"{PREDICTIONS_PATH} not found. "
        "Experiment 009 evaluation must complete first."
    )

if not TEST_PROVENANCE_PATH.exists():
    raise FileNotFoundError(
        f"{TEST_PROVENANCE_PATH} not found. "
        "Experiment 001 test provenance is required."
    )

pred_rows = pq.ParquetFile(
    PREDICTIONS_PATH
).metadata.num_rows

if pred_rows != EXPECTED_TEST_ROWS:
    raise ValueError(
        f"Unexpected Experiment 009 prediction row count: "
        f"{pred_rows} != {EXPECTED_TEST_ROWS}"
    )

predictions = pd.read_parquet(
    PREDICTIONS_PATH,
    columns=[
        "row_index",
        "true_label",
        "predicted_label",
        "confidence"
    ]
)

provenance = pd.read_csv(
    TEST_PROVENANCE_PATH,
    usecols=[
        "Original_Attack",
        "Category"
    ]
)

if len(predictions) != EXPECTED_TEST_ROWS:
    raise ValueError(
        f"Prediction DataFrame row count mismatch: "
        f"{len(predictions)} != {EXPECTED_TEST_ROWS}"
    )

if len(provenance) != EXPECTED_TEST_ROWS:
    raise ValueError(
        f"Provenance CSV row count mismatch: "
        f"{len(provenance)} != {EXPECTED_TEST_ROWS}"
    )

if len(predictions) != len(provenance):
    raise ValueError(
        f"Prediction and provenance row counts differ: "
        f"{len(predictions)} != {len(provenance)}"
    )

expected_index = np.arange(
    len(predictions),
    dtype=np.int64
)

actual_index = predictions[
    "row_index"
].to_numpy(dtype=np.int64)

if not np.array_equal(
    actual_index,
    expected_index
):
    raise ValueError(
        "Experiment 009 prediction row ordering is not sequential"
    )

prediction_true_labels = predictions[
    "true_label"
].astype(str).to_numpy()

provenance_categories = provenance[
    "Category"
].astype(str).to_numpy()

if not np.array_equal(
    prediction_true_labels,
    provenance_categories
):
    mismatches = np.flatnonzero(
        prediction_true_labels != provenance_categories
    )

    first_mismatch = int(mismatches[0])

    raise ValueError(
        "Experiment 009 true labels do not align with "
        "Experiment 001 test ordering. "
        f"First mismatch at row {first_mismatch}: "
        f"prediction true_label="
        f"{prediction_true_labels[first_mismatch]}, "
        f"provenance Category="
        f"{provenance_categories[first_mismatch]}"
    )

original_attacks = provenance[
    "Original_Attack"
].astype(str).to_numpy()

observed_attacks = sorted(
    set(original_attacks)
)

unknown_attacks = sorted(
    set(observed_attacks).difference(
        EXPECTED_ATTACKS
    )
)

if unknown_attacks:
    raise ValueError(
        f"Unexpected attack labels found: "
        f"{unknown_attacks}"
    )

predicted_labels = predictions[
    "predicted_label"
].astype(str).to_numpy()

confidences = predictions[
    "confidence"
].to_numpy(dtype=np.float64)

if not np.isfinite(confidences).all():
    raise ValueError(
        "Non-finite confidence values detected"
    )

if np.any(confidences < 0.0) or np.any(
    confidences > 1.0
):
    raise ValueError(
        "Prediction confidence values outside [0, 1]"
    )

results = []

for attack in EXPECTED_ATTACKS:
    mask = original_attacks == attack
    support = int(np.sum(mask))
    expected_category = CATEGORY_MAP[attack]

    if support == 0:
        results.append(
            {
                "attack_family": attack,
                "category": expected_category,
                "support": 0,
                "correct_category_predictions": 0,
                "category_recall": None,
                "mean_confidence": None
            }
        )

        continue

    family_predictions = predicted_labels[mask]
    family_confidences = confidences[mask]

    correct = int(
        np.sum(
            family_predictions
            == expected_category
        )
    )

    category_recall = float(
        correct / support
    )

    mean_confidence = float(
        np.mean(family_confidences)
    )

    results.append(
        {
            "attack_family": attack,
            "category": expected_category,
            "support": support,
            "correct_category_predictions": correct,
            "category_recall": category_recall,
            "mean_confidence": mean_confidence
        }
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

output = {
    "experiment_id": EXPERIMENT_ID,
    "model_source": "experiment_009_hybrid_cnn_lstm",
    "evaluation_type": (
        "original_attack_family_stratified_category_recall"
    ),
    "test_rows": int(len(predictions)),
    "prediction_rows_verified": int(pred_rows),
    "provenance_rows_verified": int(
        len(provenance)
    ),
    "row_alignment_verified": True,
    "observed_test_attack_families": observed_attacks,
    "attack_family_metrics": results,
    "important_interpretation": (
        "The classifier predicts four broad categories: "
        "Benign, DDoS, Recon, and Web. Per-attack results "
        "measure broad-category recall stratified by the "
        "original CICIoT2023 attack family. They are not "
        "eight-class attack-family predictions. Attack "
        "families with zero test support are reported with "
        "null recall and confidence rather than zero recall."
    )
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        indent=2
    )

config = {
    "experiment_id": EXPERIMENT_ID,
    "status": "completed",
    "synthetic_metrics": False,
    "mock_metrics": False,
    "predictions_source": str(
        PREDICTIONS_PATH
    ),
    "provenance_source": str(
        TEST_PROVENANCE_PATH
    ),
    "prediction_rows": int(
        len(predictions)
    ),
    "provenance_rows": int(
        len(provenance)
    ),
    "expected_test_rows": EXPECTED_TEST_ROWS,
    "row_alignment_verified": True,
    "alignment_method": (
        "sequential row_index verification plus exact "
        "row-by-row equality between Experiment 009 "
        "true_label and Experiment 001 Category"
    ),
    "classifier_output_classes": [
        "Benign",
        "DDoS",
        "Recon",
        "Web"
    ],
    "original_attack_families": EXPECTED_ATTACKS,
    "observed_test_attack_families": observed_attacks,
    "zero_support_policy": (
        "category_recall and mean_confidence are null"
    )
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
print("EXPERIMENT 005 — PER-ATTACK EVALUATION")
print()

print(
    f"Prediction rows: {len(predictions):,}"
)

print(
    f"Provenance rows: {len(provenance):,}"
)

print(
    "Row alignment: PASS"
)

print()

print(
    results_df.to_string(
        index=False
    )
)

print()

print(
    f"Saved: {OUTPUT_CSV}"
)

print(
    f"Saved: {OUTPUT_JSON}"
)

print(
    f"Saved: {CONFIG_PATH}"
)

print()

print(
    "Experiment 005 completed successfully."
)