import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_ID = "008"
EXPERIMENT_NAME = "Model-Driven Mitigation Policy Evaluation"

INPUT_PATH = Path(
    "runs/experiment_009_hybrid_cnn_lstm/test_predictions.parquet"
)

OUTPUT_DIR = Path(
    "runs/experiment_008_mitigation"
)

DECISIONS_PATH = OUTPUT_DIR / "mitigation_decisions.csv"
THRESHOLD_PATH = OUTPUT_DIR / "threshold_analysis.csv"
METRICS_PATH = OUTPUT_DIR / "mitigation_metrics.json"
SUMMARY_PATH = OUTPUT_DIR / "experiment_summary.json"
CONFIG_PATH = OUTPUT_DIR / "config.json"

EXPECTED_ROWS = 1_773_902
EXPECTED_INPUT_SHA256 = (
    "d8422a95daceb3431f7442f7a9eeab267982e5a1bdc9f6c52105d88b369b8a24"
)

BENIGN_LABEL = "Benign"
ATTACK_LABELS = {"DDoS", "Recon", "Web"}

OPERATING_THRESHOLD = 0.85

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    0.99,
]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def safe_rate(numerator, denominator):
    if denominator == 0:
        return None

    return numerator / denominator


def validate_input(df):
    required_columns = {
        "row_index",
        "true_index",
        "true_label",
        "predicted_index",
        "predicted_label",
        "confidence",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, found {len(df)}"
        )

    if df[list(required_columns)].isna().any().any():
        raise ValueError(
            "Prediction artifact contains missing values"
        )

    row_index = pd.to_numeric(
        df["row_index"],
        errors="raise"
    ).astype("int64").reset_index(drop=True)

    expected_index = pd.Series(
        range(EXPECTED_ROWS),
        dtype="int64"
    )

    if not row_index.equals(expected_index):
        raise ValueError(
            "row_index is not exactly 0..EXPECTED_ROWS-1"
        )

    if row_index.duplicated().any():
        raise ValueError(
            "row_index contains duplicates"
        )

    allowed_labels = {BENIGN_LABEL}.union(
        ATTACK_LABELS
    )

    true_labels = set(
        df["true_label"].astype(str).unique()
    )

    predicted_labels = set(
        df["predicted_label"].astype(str).unique()
    )

    unexpected_true = true_labels.difference(
        allowed_labels
    )

    unexpected_predicted = predicted_labels.difference(
        allowed_labels
    )

    if unexpected_true:
        raise ValueError(
            f"Unexpected true labels: "
            f"{sorted(unexpected_true)}"
        )

    if unexpected_predicted:
        raise ValueError(
            f"Unexpected predicted labels: "
            f"{sorted(unexpected_predicted)}"
        )

    confidence = pd.to_numeric(
        df["confidence"],
        errors="raise"
    ).astype(float)

    if not confidence.map(math.isfinite).all():
        raise ValueError(
            "Confidence contains non-finite values"
        )

    if (
        (confidence < 0.0)
        | (confidence > 1.0)
    ).any():
        raise ValueError(
            "Confidence outside [0,1]"
        )

    df = df.copy()
    df["row_index"] = row_index
    df["true_label"] = df[
        "true_label"
    ].astype(str)
    df["predicted_label"] = df[
        "predicted_label"
    ].astype(str)
    df["confidence"] = confidence

    return df


def evaluate_threshold(df, threshold):
    true_benign = (
        df["true_label"] == BENIGN_LABEL
    )

    true_malicious = ~true_benign

    predicted_malicious = (
        df["predicted_label"] != BENIGN_LABEL
    )

    selected = (
        predicted_malicious
        & (df["confidence"] >= threshold)
    )

    malicious_total = int(
        true_malicious.sum()
    )

    benign_total = int(
        true_benign.sum()
    )

    malicious_blocked = int(
        (true_malicious & selected).sum()
    )

    malicious_allowed = int(
        (true_malicious & ~selected).sum()
    )

    benign_blocked = int(
        (true_benign & selected).sum()
    )

    benign_allowed = int(
        (true_benign & ~selected).sum()
    )

    total_blocked = int(
        selected.sum()
    )

    total_allowed = int(
        (~selected).sum()
    )

    result = {
        "threshold": float(threshold),
        "total_observations": int(len(df)),
        "total_blocked": total_blocked,
        "total_allowed": total_allowed,
        "malicious_total": malicious_total,
        "malicious_blocked": malicious_blocked,
        "malicious_allowed": malicious_allowed,
        "malicious_block_rate": safe_rate(
            malicious_blocked,
            malicious_total
        ),
        "malicious_allow_rate": safe_rate(
            malicious_allowed,
            malicious_total
        ),
        "benign_total": benign_total,
        "benign_blocked": benign_blocked,
        "benign_allowed": benign_allowed,
        "benign_block_rate": safe_rate(
            benign_blocked,
            benign_total
        ),
        "benign_allow_rate": safe_rate(
            benign_allowed,
            benign_total
        ),
    }

    for label in [
        "DDoS",
        "Recon",
        "Web",
    ]:
        mask = (
            df["true_label"] == label
        )

        support = int(
            mask.sum()
        )

        blocked = int(
            (mask & selected).sum()
        )

        allowed = int(
            (mask & ~selected).sum()
        )

        result[
            f"{label}_support"
        ] = support

        result[
            f"{label}_blocked"
        ] = blocked

        result[
            f"{label}_allowed"
        ] = allowed

        result[
            f"{label}_block_rate"
        ] = safe_rate(
            blocked,
            support
        )

    return result, selected


def validate_reconciliation(metrics):
    if (
        metrics["total_blocked"]
        + metrics["total_allowed"]
        != metrics["total_observations"]
    ):
        raise RuntimeError(
            "Total BLOCK/ALLOW reconciliation failed"
        )

    if (
        metrics["malicious_blocked"]
        + metrics["malicious_allowed"]
        != metrics["malicious_total"]
    ):
        raise RuntimeError(
            "Malicious reconciliation failed"
        )

    if (
        metrics["benign_blocked"]
        + metrics["benign_allowed"]
        != metrics["benign_total"]
    ):
        raise RuntimeError(
            "Benign reconciliation failed"
        )

    if (
        metrics["malicious_total"]
        + metrics["benign_total"]
        != metrics["total_observations"]
    ):
        raise RuntimeError(
            "Population reconciliation failed"
        )

    for label in [
        "DDoS",
        "Recon",
        "Web",
    ]:
        if (
            metrics[f"{label}_blocked"]
            + metrics[f"{label}_allowed"]
            != metrics[f"{label}_support"]
        ):
            raise RuntimeError(
                f"{label} reconciliation failed"
            )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Exp009 prediction artifact: "
            f"{INPUT_PATH}"
        )

    if not INPUT_PATH.is_file():
        raise ValueError(
            f"Prediction source is not a regular file: "
            f"{INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    input_sha256_before = sha256_file(
        INPUT_PATH
    )

    if (
        input_sha256_before
        != EXPECTED_INPUT_SHA256
    ):
        raise RuntimeError(
            "Exp009 prediction SHA256 does not match "
            "the previously verified artifact. "
            f"Expected {EXPECTED_INPUT_SHA256}, "
            f"found {input_sha256_before}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    df = validate_input(
        df
    )

    true_distribution = {
        str(key): int(value)
        for key, value in
        df["true_label"]
        .value_counts()
        .sort_index()
        .items()
    }

    predicted_distribution = {
        str(key): int(value)
        for key, value in
        df["predicted_label"]
        .value_counts()
        .sort_index()
        .items()
    }

    threshold_results = []
    operating_selected = None
    operating_metrics = None

    for threshold in THRESHOLDS:
        result, selected = evaluate_threshold(
            df,
            threshold
        )

        validate_reconciliation(
            result
        )

        threshold_results.append(
            result
        )

        if math.isclose(
            threshold,
            OPERATING_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12
        ):
            operating_selected = (
                selected.copy()
            )

            operating_metrics = (
                result.copy()
            )

    if operating_selected is None:
        raise RuntimeError(
            "Operating threshold was not evaluated"
        )

    if operating_metrics is None:
        raise RuntimeError(
            "Operating metrics were not generated"
        )

    decisions = pd.DataFrame(
        {
            "row_index": df[
                "row_index"
            ].astype("int64"),
            "true_index": df[
                "true_index"
            ],
            "true_label": df[
                "true_label"
            ],
            "predicted_index": df[
                "predicted_index"
            ],
            "predicted_label": df[
                "predicted_label"
            ],
            "confidence": df[
                "confidence"
            ].astype(float),
            "threshold": (
                OPERATING_THRESHOLD
            ),
            "mitigation_selected": (
                operating_selected
            ),
            "decision": (
                operating_selected.map(
                    {
                        True: "BLOCK",
                        False: "ALLOW",
                    }
                )
            ),
        }
    )

    if len(decisions) != EXPECTED_ROWS:
        raise RuntimeError(
            "Decision output row count mismatch"
        )

    if (
        decisions["row_index"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Decision output contains duplicate row_index"
        )

    decisions.to_csv(
        DECISIONS_PATH,
        index=False
    )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    threshold_df.to_csv(
        THRESHOLD_PATH,
        index=False,
        quoting=csv.QUOTE_MINIMAL
    )

    input_sha256_after = sha256_file(
        INPUT_PATH
    )

    if (
        input_sha256_before
        != input_sha256_after
    ):
        raise RuntimeError(
            "Exp009 prediction artifact changed "
            "during Exp008"
        )

    decision_sha256 = sha256_file(
        DECISIONS_PATH
    )

    threshold_sha256 = sha256_file(
        THRESHOLD_PATH
    )

    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "operating_threshold": (
            OPERATING_THRESHOLD
        ),
        **operating_metrics,
        "true_label_distribution": (
            true_distribution
        ),
        "predicted_label_distribution": (
            predicted_distribution
        ),
        "decision_rule": (
            "BLOCK when predicted_label != Benign "
            "and confidence >= threshold; "
            "otherwise ALLOW"
        ),
        "ground_truth_used_for_decision": False,
        "ground_truth_used_for_evaluation": True,
        "prediction_source": str(
            INPUT_PATH
        ),
        "prediction_source_sha256": (
            input_sha256_before
        ),
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            metrics,
            handle,
            indent=2
        )

    metrics_sha256 = sha256_file(
        METRICS_PATH
    )

    config = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "status": "completed",
        "experiment_type": (
            "offline_model_driven_"
            "mitigation_policy_evaluation"
        ),
        "prediction_source": str(
            INPUT_PATH
        ),
        "prediction_source_format": (
            "parquet"
        ),
        "prediction_source_sha256": (
            input_sha256_before
        ),
        "expected_prediction_source_sha256": (
            EXPECTED_INPUT_SHA256
        ),
        "expected_rows": EXPECTED_ROWS,
        "rows_processed": int(
            len(df)
        ),
        "operating_threshold": (
            OPERATING_THRESHOLD
        ),
        "thresholds_evaluated": (
            THRESHOLDS
        ),
        "benign_label": (
            BENIGN_LABEL
        ),
        "attack_labels": sorted(
            ATTACK_LABELS
        ),
        "decision_rule": (
            "predicted_label != Benign "
            "and confidence >= threshold"
        ),
        "prediction_source_is_existing_exp009_output": True,
        "predictions_regenerated_by_exp008": False,
        "model_retrained_by_exp008": False,
        "synthetic_predictions": False,
        "synthetic_confidence": False,
        "synthetic_data": False,
        "mock_traffic": False,
        "sampling": False,
        "all_test_rows_processed": True,
        "ground_truth_used_for_decision": False,
        "ground_truth_used_for_evaluation": True,
        "source_ip_available": False,
        "live_feature_extraction_performed": False,
        "live_model_inference_performed": False,
        "live_enforcement_performed": False,
        "iptables_enforcement_claimed": False,
        "real_time_detection_claimed": False,
        "end_to_end_live_mitigation_claimed": False,
        "python_version": (
            sys.version
        ),
        "platform": (
            platform.platform()
        ),
        "output_files": {
            "mitigation_decisions": str(
                DECISIONS_PATH
            ),
            "threshold_analysis": str(
                THRESHOLD_PATH
            ),
            "mitigation_metrics": str(
                METRICS_PATH
            ),
            "experiment_summary": str(
                SUMMARY_PATH
            ),
            "config": str(
                CONFIG_PATH
            ),
        },
    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            config,
            handle,
            indent=2
        )

    summary = {
        "experiment_id": (
            EXPERIMENT_ID
        ),
        "experiment_name": (
            EXPERIMENT_NAME
        ),
        "status": "completed",
        "scope": (
            "Offline mitigation-policy evaluation "
            "using genuine held-out Exp009 predictions"
        ),
        "rows_processed": int(
            len(df)
        ),
        "operating_threshold": (
            OPERATING_THRESHOLD
        ),
        "operating_metrics": (
            operating_metrics
        ),
        "true_label_distribution": (
            true_distribution
        ),
        "predicted_label_distribution": (
            predicted_distribution
        ),
        "prediction_source": str(
            INPUT_PATH
        ),
        "prediction_source_sha256": (
            input_sha256_before
        ),
        "prediction_source_unchanged": True,
        "mitigation_decisions_sha256": (
            decision_sha256
        ),
        "threshold_analysis_sha256": (
            threshold_sha256
        ),
        "mitigation_metrics_sha256": (
            metrics_sha256
        ),
        "synthetic_data_generated": False,
        "synthetic_predictions_generated": False,
        "synthetic_confidence_generated": False,
        "sampling_performed": False,
        "ground_truth_used_for_decision": False,
        "ground_truth_used_for_evaluation": True,
        "live_feature_extraction_claimed": False,
        "live_model_inference_claimed": False,
        "live_enforcement_claimed": False,
        "iptables_enforcement_claimed": False,
        "end_to_end_live_mitigation_claimed": False,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2
        )

    print(
        json.dumps(
            summary,
            indent=2
        )
    )


if __name__ == "__main__":
    main()