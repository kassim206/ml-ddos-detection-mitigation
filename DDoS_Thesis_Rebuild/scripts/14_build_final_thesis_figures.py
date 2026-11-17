#!/usr/bin/env python3

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

TABLES = Path("tables")
FIGURES = Path("figures")
MATRIX = Path(
    "runs/thesis_evidence_reconciliation_v1/"
    "thesis_evidence_matrix.json"
)

EXPECTED_TABLE_MANIFEST_SHA256 = (
    "cfc3d990094b58f3ae3ac16b7577d27100205e6a5f90a1b7cdaf4bf2596847ac"
)

EXPECTED_MATRIX_SHA256 = (
    "fb37520b0b96da5d5d21f46bb617c97a604965e88c0a428af82e94d213b826ac"
)

EXPECTED_TABLES = {
    "table_01_dataset_a_composition_split.csv":
        "283d28cc382cc3ffc0143a2bc4419656aa03abde5007faf432b40efadc58a953",
    "table_02_dataset_a_model_performance.csv":
        "c781e4a96a9c8e99a6d5f5dd0f44fbd0225234d6b806a82cae8e8e974f041c60",
    "table_03_attack_family_performance.csv":
        "413a4284d74858fd9bacdd251f26f583ce637a78000b2b45840697c3a094d481",
    "table_04_cpu_batch_inference_runtime.csv":
        "ddd8ea2415c3573cf23917461f70d02b4981d7f18e8871c476c13e98ffa5d92c",
    "table_05_controlled_http_service_baseline.csv":
        "b5f1430208bc3acd18cbac207334679226140533b2f6e38c2ba0d21744149971",
    "table_06_offline_mitigation_policy.csv":
        "e9b246f0dd1736016027ca9b796a9975d121a92ca2822c8400057d38b713c8be",
    "table_07_dataset_b_heldout_detection.csv":
        "2d21faa3ca7531d0fedacbd6dbf8f921579f2c4caba2ad20fa3b89a2873472bb",
    "table_08_live_detection_runtime.csv":
        "7380c094222f36ee6a2c6f4057a4efe891f32ba586ef995fade8c245f3882836",
    "table_09_live_mitigation_service_impact.csv":
        "a26865229f508f2cb0674fa6ea2f6ffe27cfefde1dec9eb11b157cf07f71fac1"
}

EXPECTED_FIGURES = [
    "figure_01_dataset_a_partition.png",
    "figure_02_dataset_a_model_performance.png",
    "figure_03_attack_family_performance.png",
    "figure_04_cpu_inference_runtime.png",
    "figure_05_http_service_baseline.png",
    "figure_06_offline_mitigation_policy.png",
    "figure_07_dataset_b_heldout_detection.png",
    "figure_08_live_detection_runtime.png",
    "figure_09_live_mitigation_service.png"
]


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def read_csv(name):
    with (TABLES / name).open(
        newline="",
        encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))


def verify_inputs():
    manifest = TABLES / "TABLES_SHA256SUMS"

    if not manifest.exists():
        raise RuntimeError(
            "Frozen table manifest missing"
        )

    if sha256(manifest) != EXPECTED_TABLE_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen table manifest hash mismatch"
        )

    for name, expected in EXPECTED_TABLES.items():
        path = TABLES / name

        if not path.exists():
            raise RuntimeError(
                f"Missing frozen table: {name}"
            )

        actual = sha256(path)

        if actual != expected:
            raise RuntimeError(
                f"Frozen table hash mismatch: {name}"
            )

    if not MATRIX.exists():
        raise RuntimeError(
            "Frozen evidence matrix missing"
        )

    if sha256(MATRIX) != EXPECTED_MATRIX_SHA256:
        raise RuntimeError(
            "Frozen evidence matrix hash mismatch"
        )


def configure():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 350,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.7
    })


def finish(
    fig,
    name,
    footer=None,
    bottom=0.12,
    left=0.12,
    right=0.97,
    top=0.90
):
    if footer:
        fig.text(
            left,
            0.025,
            footer,
            ha="left",
            va="bottom",
            fontsize=7.8,
            style="italic",
            wrap=True
        )

    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top
    )

    path = FIGURES / name

    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    return path


def fmt_count(x):
    return f"{int(x):,}"


def fmt_rate(x):
    return f"{x * 100:.1f}%"


verify_inputs()
configure()

matrix = json.loads(
    MATRIX.read_text(
        encoding="utf-8"
    )
)

by_exp = {
    x["experiment"]: x
    for x in matrix["experiments"]
}

FIGURES.mkdir(
    parents=True,
    exist_ok=True
)

for path in FIGURES.glob(
    "figure_*.png"
):
    path.unlink()

t1 = {
    row["item"]: row["value"]
    for row in read_csv(
        "table_01_dataset_a_composition_split.csv"
    )
}

total = int(
    t1["Total rows"]
)

split_names = [
    "Training",
    "Validation",
    "Test"
]

split_counts = np.array(
    [
        int(t1["Training rows"]),
        int(t1["Validation rows"]),
        int(t1["Test rows"])
    ],
    dtype=int
)

if split_counts.sum() != total:
    raise RuntimeError(
        "Figure 01 split reconciliation failed"
    )

split_pct = (
    split_counts
    / total
    * 100
)

fig, ax = plt.subplots(
    figsize=(9.0, 4.8)
)

y = np.arange(
    len(split_names)
)

ax.barh(
    y,
    split_pct,
    height=0.48
)

ax.set_yticks(
    y,
    split_names
)

ax.set_xlim(
    0,
    80
)

ax.set_xlabel(
    "Share of Dataset A (%)"
)

ax.set_title(
    "Dataset A Experimental Partition",
    loc="left",
    pad=14
)

ax.grid(
    axis="x"
)

for yi, pct, count in zip(
    y,
    split_pct,
    split_counts
):
    ax.text(
        pct + 1.0,
        yi,
        f"{pct:.1f}%  |  {count:,} rows",
        va="center",
        ha="left",
        fontsize=9
    )

ax.invert_yaxis()

footer = (
    f"Total: {total:,} rows; "
    f"{int(t1['Feature count'])} features; "
    f"{int(t1['Source feature CSV files'])} source feature CSV files. "
    f"Split construction used {int(t1['Whole-file allocations'])} whole-file "
    f"allocations and {int(t1['Row-level partitioned files'])} row-level "
    f"partitioned files."
)

finish(
    fig,
    EXPECTED_FIGURES[0],
    footer=footer,
    bottom=0.17,
    left=0.14
)

t2 = read_csv(
    "table_02_dataset_a_model_performance.csv"
)

models = [
    row["model"]
    for row in t2
]

accuracy = np.array(
    [
        float(row["accuracy"])
        for row in t2
    ]
)

macro_f1 = np.array(
    [
        float(row["macro_f1"])
        for row in t2
    ]
)

fig, ax = plt.subplots(
    figsize=(9.5, 5.4)
)

y = np.arange(
    len(models)
)

for yi, f1, acc in zip(
    y,
    macro_f1,
    accuracy
):
    ax.hlines(
        yi,
        f1,
        acc,
        linewidth=2
    )

ax.scatter(
    accuracy,
    y,
    s=70,
    marker="o",
    label="Accuracy",
    zorder=3
)

ax.scatter(
    macro_f1,
    y,
    s=70,
    marker="D",
    label="Macro-F1",
    zorder=3
)

ax.set_yticks(
    y,
    models
)

ax.set_xlim(
    0,
    1.08
)

ax.xaxis.set_major_formatter(
    PercentFormatter(
        xmax=1.0
    )
)

ax.set_xlabel(
    "Score"
)

ax.set_title(
    "Dataset A Model Performance",
    loc="left",
    pad=14
)

ax.grid(
    axis="x"
)

for yi, acc, f1 in zip(
    y,
    accuracy,
    macro_f1
):
    ax.annotate(
        f"{acc:.3f}",
        (acc, yi),
        xytext=(6, -11),
        textcoords="offset points",
        fontsize=8
    )

    ax.annotate(
        f"{f1:.3f}",
        (f1, yi),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8
    )

ax.legend(
    frameon=False,
    loc="lower right"
)

ax.invert_yaxis()

footer = (
    "Random Forest, XGBoost and Hybrid CNN-LSTM values are held-out test results. "
    "CNN-Dense is validation-only and must not be interpreted as a held-out-test "
    "comparison. The accuracy–macro-F1 gap reflects class-imbalanced performance."
)

finish(
    fig,
    EXPECTED_FIGURES[1],
    footer=footer,
    bottom=0.18,
    left=0.19
)

t3 = read_csv(
    "table_03_attack_family_performance.csv"
)

family_names = [
    row["attack_family"]
    for row in t3
]

supports = np.array(
    [
        int(row["support"])
        for row in t3
    ],
    dtype=int
)

recalls = [
    (
        None
        if row["category_recall"] == ""
        else float(
            row["category_recall"]
        )
    )
    for row in t3
]

confidence = [
    (
        None
        if row["mean_confidence"] == ""
        else float(
            row["mean_confidence"]
        )
    )
    for row in t3
]

fig, ax = plt.subplots(
    figsize=(10.2, 6.4)
)

y = np.arange(
    len(family_names)
)

for yi, recall, support in zip(
    y,
    recalls,
    supports
):
    if recall is not None:
        ax.hlines(
            yi,
            0,
            recall,
            linewidth=1.8
        )

        ax.scatter(
            recall,
            yi,
            s=70,
            zorder=3
        )

        ax.annotate(
            f"{recall:.3f}  |  n={support:,}",
            (recall, yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.2
        )
    else:
        ax.scatter(
            0.02,
            yi,
            s=45,
            marker="x"
        )

        ax.text(
            0.055,
            yi,
            "Not evaluated  |  n=0",
            va="center",
            fontsize=8.2,
            style="italic"
        )

ax.set_yticks(
    y,
    family_names
)

ax.set_xlim(
    0,
    1.16
)

ax.xaxis.set_major_formatter(
    PercentFormatter(
        xmax=1.0
    )
)

ax.set_xlabel(
    "Category recall"
)

ax.set_title(
    "Hybrid Model Performance by Original Attack Family",
    loc="left",
    pad=14
)

ax.grid(
    axis="x"
)

ax.invert_yaxis()

evaluated_conf = [
    x
    for x in confidence
    if x is not None
]

if len(evaluated_conf) != 5:
    raise RuntimeError(
        "Figure 03 evaluated-family count mismatch"
    )

footer = (
    "All eight original attack families are represented. Families with zero held-out "
    "support are explicitly marked as not evaluated rather than assigned zero recall. "
    "Displayed recall is category recall from the frozen Exp005/Exp009 evidence."
)

finish(
    fig,
    EXPECTED_FIGURES[2],
    footer=footer,
    bottom=0.16,
    left=0.23
)

t4 = read_csv(
    "table_04_cpu_batch_inference_runtime.csv"
)

runtime_order = [
    "RandomForest",
    "XGBoost",
    "CNN-Dense",
    "Hybrid-CNN-LSTM"
]

runtime_label = {
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "CNN-Dense": "CNN-Dense",
    "Hybrid-CNN-LSTM": "Hybrid CNN-LSTM"
}

runtime_by = {
    row["model"]: row
    for row in t4
}

means = np.array(
    [
        float(
            runtime_by[m][
                "mean_total_seconds"
            ]
        )
        for m in runtime_order
    ]
)

stds = np.array(
    [
        float(
            runtime_by[m][
                "std_total_seconds"
            ]
        )
        for m in runtime_order
    ]
)

throughput = np.array(
    [
        float(
            runtime_by[m][
                "throughput_samples_per_second"
            ]
        )
        for m in runtime_order
    ]
)

fig, ax = plt.subplots(
    figsize=(9.7, 5.4)
)

y = np.arange(
    len(runtime_order)
)

ax.errorbar(
    means,
    y,
    xerr=stds,
    fmt="o",
    markersize=7,
    capsize=5,
    linewidth=1.4
)

ax.set_yticks(
    y,
    [
        runtime_label[m]
        for m in runtime_order
    ]
)

limit = max(
    means + stds
) * 1.35

ax.set_xlim(
    0,
    limit
)

ax.set_xlabel(
    "Complete test-split CPU inference time (s)"
)

ax.set_title(
    "CPU Batched Inference Runtime",
    loc="left",
    pad=14
)

ax.grid(
    axis="x"
)

for yi, mean, std, rate in zip(
    y,
    means,
    stds,
    throughput
):
    ax.annotate(
        f"{mean:.2f} ± {std:.2f} s  |  {rate:,.0f} samples/s",
        (mean + std, yi),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.2
    )

ax.invert_yaxis()

rows_per_repeat = int(
    runtime_by[
        "RandomForest"
    ][
        "rows_per_repeat"
    ]
)

footer = (
    f"Each point is the measured mean across three complete CPU-batched inference "
    f"repeats over {rows_per_repeat:,} test rows; horizontal intervals show the "
    "recorded population standard deviation. These are batch-runtime measurements, "
    "not per-request latency."
)

finish(
    fig,
    EXPECTED_FIGURES[3],
    footer=footer,
    bottom=0.18,
    left=0.21
)

t5 = {
    row["metric"]: row["value"]
    for row in read_csv(
        "table_05_controlled_http_service_baseline.csv"
    )
}

attempted = int(
    t5["Requests attempted"]
)

successful = int(
    t5[
        "Transport successful requests"
    ]
)

elapsed = float(
    t5[
        "Total elapsed seconds"
    ]
)

throughput_value = float(
    t5[
        "Throughput requests per second"
    ]
)

mean_ms = float(
    t5[
        "Mean response time ms"
    ]
)

p95_ms = float(
    t5[
        "P95 response time ms"
    ]
)

if successful > attempted:
    raise RuntimeError(
        "Figure 05 request accounting invalid"
    )

fig, ax = plt.subplots(
    figsize=(9.5, 5.0)
)

metric_names = [
    "Transport success",
    "Elapsed time",
    "Throughput",
    "Mean response",
    "P95 response"
]

metric_values = [
    f"{successful:,}/{attempted:,}",
    f"{elapsed:.3f} s",
    f"{throughput_value:.2f} req/s",
    f"{mean_ms:.2f} ms",
    f"{p95_ms:.2f} ms"
]

x = np.arange(
    len(metric_names)
)

ax.set_xlim(
    -0.5,
    len(metric_names) - 0.5
)

ax.set_ylim(
    0,
    1
)

ax.axis("off")

ax.set_title(
    "Controlled HTTP Service Baseline",
    loc="left",
    pad=14
)

for xi, name, value in zip(
    x,
    metric_names,
    metric_values
):
    ax.text(
        xi,
        0.61,
        value,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="semibold"
    )

    ax.text(
        xi,
        0.39,
        name,
        ha="center",
        va="center",
        fontsize=8.5
    )

for xi in np.arange(
    0.5,
    len(metric_names) - 0.5,
    1
):
    ax.axvline(
        xi,
        ymin=0.25,
        ymax=0.75,
        linewidth=0.7,
        alpha=0.3
    )

footer = (
    "Exp007 controlled local HTTP baseline: 1,000 genuine GET requests at concurrency "
    "10, all transport- and HTTP-successful. This experiment did not measure DDoS "
    "detection, ML inference, or mitigation."
)

finish(
    fig,
    EXPECTED_FIGURES[4],
    footer=footer,
    bottom=0.18,
    left=0.08,
    right=0.98
)

t6 = {
    row["metric"]: row["value"]
    for row in read_csv(
        "table_06_offline_mitigation_policy.csv"
    )
}

policy_names = [
    "DDoS",
    "Malicious overall",
    "Benign",
    "Web",
    "Recon"
]

policy_rates = np.array(
    [
        float(
            t6[
                "DDoS block rate"
            ]
        ),
        float(
            t6[
                "Malicious block rate"
            ]
        ),
        float(
            t6[
                "Benign block rate"
            ]
        ),
        float(
            t6[
                "Web block rate"
            ]
        ),
        float(
            t6[
                "Recon block rate"
            ]
        )
    ]
)

threshold_value = float(
    t6[
        "Decision threshold"
    ]
)

fig, ax = plt.subplots(
    figsize=(9.4, 5.5)
)

y = np.arange(
    len(policy_names)
)

ax.barh(
    y,
    policy_rates,
    height=0.45
)

ax.set_yticks(
    y,
    policy_names
)

ax.set_xlim(
    0,
    1.12
)

ax.xaxis.set_major_formatter(
    PercentFormatter(
        xmax=1.0
    )
)

ax.set_xlabel(
    "Blocked observations"
)

ax.set_title(
    "Offline Mitigation Policy Block Rates",
    loc="left",
    pad=14
)

ax.grid(
    axis="x"
)

for yi, value in zip(
    y,
    policy_rates
):
    ax.text(
        value + 0.012,
        yi,
        fmt_rate(value),
        va="center",
        fontsize=8.5
    )

ax.invert_yaxis()

footer = (
    f"Exp008 applied a frozen decision threshold of {threshold_value:.2f} to "
    f"{int(t6['Total observations']):,} offline test observations. These are offline "
    "policy outcomes, not measurements of live HTTP enforcement or network-layer blocking."
)

finish(
    fig,
    EXPECTED_FIGURES[5],
    footer=footer,
    bottom=0.18,
    left=0.20
)

t7 = {
    row["metric"]: row["value"]
    for row in read_csv(
        "table_07_dataset_b_heldout_detection.csv"
    )
}

cm10 = np.array(
    json.loads(
        t7[
            "Confusion matrix"
        ]
    ),
    dtype=int
)

if cm10.shape != (2, 2):
    raise RuntimeError(
        "Figure 07 confusion matrix shape invalid"
    )

heldout_windows = int(
    t7[
        "Held-out windows"
    ]
)

if cm10.sum() != heldout_windows:
    raise RuntimeError(
        "Figure 07 confusion matrix total mismatch"
    )

fig, ax = plt.subplots(
    figsize=(7.5, 6.4)
)

image = ax.imshow(
    cm10,
    cmap="Blues",
    interpolation="nearest"
)

cbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.05
)

cbar.set_label(
    "Window count"
)

ax.set_xticks(
    [0, 1],
    [
        "Benign",
        "High-rate HTTP"
    ]
)

ax.set_yticks(
    [0, 1],
    [
        "Benign",
        "High-rate HTTP"
    ]
)

ax.set_xlabel(
    "Predicted class"
)

ax.set_ylabel(
    "True class"
)

ax.set_title(
    "Dataset B Held-Out Detection",
    loc="left",
    pad=14
)

mid = cm10.max() / 2

for i in range(2):
    for j in range(2):
        value = cm10[i, j]

        ax.text(
            j,
            i,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="semibold",
            color=(
                "white"
                if value > mid
                else "black"
            )
        )

metric_text = (
    f"Accuracy {float(t7['Accuracy']):.3f}     "
    f"Balanced accuracy {float(t7['Balanced accuracy']):.3f}     "
    f"Macro-F1 {float(t7['Macro-F1']):.3f}"
)

fig.text(
    0.50,
    0.115,
    metric_text,
    ha="center",
    fontsize=9
)

footer = (
    f"Exp010 held-out Dataset B result: {heldout_windows} non-overlapping 100-packet "
    f"windows, selected logistic-regression model, threshold {float(t7['Threshold']):.2f}. "
    "This is a controlled-lab binary benign versus predefined high-rate HTTP result."
)

finish(
    fig,
    EXPECTED_FIGURES[6],
    footer=footer,
    bottom=0.22,
    left=0.14,
    right=0.91
)

t8 = read_csv(
    "table_08_live_detection_runtime.csv"
)

pooled_rows = [
    row
    for row in t8
    if row["scope"] == "pooled_overall"
]

condition_rows = [
    row
    for row in t8
    if row["scope"] == "condition_timing"
]

if len(pooled_rows) != 1:
    raise RuntimeError(
        "Figure 08 pooled-row count invalid"
    )

if len(condition_rows) != 3:
    raise RuntimeError(
        "Figure 08 timing-row count invalid"
    )

pooled = pooled_rows[0]

condition_by = {
    row["condition"]: row
    for row in condition_rows
}

condition_order = [
    "benign_with_detection_and_mitigation",
    "attack_detection_without_mitigation",
    "attack_detection_with_mitigation"
]

condition_labels = [
    "Benign",
    "Attack · mitigation off",
    "Attack · mitigation on"
]

feature_ms = np.array(
    [
        float(
            condition_by[c][
                "feature_to_inference_mean_seconds"
            ]
        ) * 1000
        for c in condition_order
    ]
)

inference_ms = np.array(
    [
        float(
            condition_by[c][
                "inference_mean_seconds"
            ]
        ) * 1000
        for c in condition_order
    ]
)

observation_ms = np.array(
    [
        float(
            condition_by[c][
                "window_observation_mean_seconds"
            ]
        ) * 1000
        for c in condition_order
    ]
)

enforcement_ms = (
    float(
        condition_by[
            "attack_detection_with_mitigation"
        ][
            "decision_to_enforcement_mean_seconds"
        ]
    )
    * 1000
)

fig = plt.figure(
    figsize=(10.5, 7.0)
)

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[
        1.0,
        1.35
    ],
    hspace=0.42
)

ax1 = fig.add_subplot(
    gs[0]
)

ax2 = fig.add_subplot(
    gs[1]
)

fig.suptitle(
    "Controlled-Lab Live Detection and Runtime",
    x=0.10,
    y=0.965,
    ha="left",
    fontsize=13,
    fontweight="semibold"
)

metric_names = [
    "Accuracy",
    "Balanced\naccuracy",
    "Precision",
    "Recall",
    "F1",
    "Macro-F1"
]

metric_values = np.array(
    [
        float(pooled["accuracy"]),
        float(pooled["balanced_accuracy"]),
        float(pooled["precision"]),
        float(pooled["recall"]),
        float(pooled["f1"]),
        float(pooled["macro_f1"])
    ]
)

x = np.arange(
    len(metric_names)
)

ax1.scatter(
    x,
    metric_values,
    s=65
)

ax1.vlines(
    x,
    0,
    metric_values,
    linewidth=1.2
)

ax1.set_xticks(
    x,
    metric_names
)

ax1.set_ylim(
    0,
    1.08
)

ax1.yaxis.set_major_formatter(
    PercentFormatter(
        xmax=1.0
    )
)

ax1.set_ylabel(
    "Pooled score"
)

ax1.grid(
    axis="y"
)

for xi, value in zip(
    x,
    metric_values
):
    ax1.text(
        xi,
        value + 0.018,
        f"{value:.3f}",
        ha="center",
        fontsize=8
    )

y = np.arange(
    len(condition_order)
)

ax2.hlines(
    y,
    inference_ms,
    feature_ms,
    linewidth=2
)

ax2.scatter(
    inference_ms,
    y,
    s=60,
    marker="s",
    label="Inference execution"
)

ax2.scatter(
    feature_ms,
    y,
    s=65,
    marker="o",
    label="Feature completion → inference completion"
)

ax2.set_yticks(
    y,
    condition_labels
)

ax2.set_xlabel(
    "Mean measured runtime (ms)"
)

ax2.set_xlim(
    0,
    max(
        feature_ms.max(),
        enforcement_ms
    ) * 1.55
)

ax2.grid(
    axis="x"
)

for yi, infer, feature in zip(
    y,
    inference_ms,
    feature_ms
):
    ax2.annotate(
        f"{infer:.3f}",
        (infer, yi),
        xytext=(-6, -13),
        textcoords="offset points",
        ha="right",
        fontsize=7.8
    )

    ax2.annotate(
        f"{feature:.3f}",
        (feature, yi),
        xytext=(6, 7),
        textcoords="offset points",
        fontsize=7.8
    )

ax2.axvline(
    enforcement_ms,
    linestyle="--",
    linewidth=1.1,
    label=(
        "Decision → enforcement mean "
        f"{enforcement_ms:.3f} ms"
    )
)

ax2.legend(
    frameon=False,
    loc="lower right",
    fontsize=7.8
)

ax2.invert_yaxis()

observation_text = (
    "Mean 100-packet observation duration: "
    f"benign {observation_ms[0]:.1f} ms; "
    f"attack/mitigation-off {observation_ms[1]:.1f} ms; "
    f"attack/mitigation-on {observation_ms[2]:.1f} ms."
)

fig.text(
    0.10,
    0.025,
    (
        f"Pooled detection uses {int(pooled['observed_windows']):,} windows nested "
        f"within nine experimental runs. {observation_text} Runtime intervals are "
        "not request latency or end-to-end detection latency."
    ),
    ha="left",
    va="bottom",
    fontsize=7.7,
    style="italic",
    wrap=True
)

fig.subplots_adjust(
    left=0.20,
    right=0.96,
    bottom=0.16,
    top=0.89
)

fig.savefig(
    FIGURES / EXPECTED_FIGURES[7],
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

t9 = read_csv(
    "table_09_live_mitigation_service_impact.csv"
)

service_by = {
    row["condition"]: row
    for row in t9
}

service_order = [
    "benign_with_detection_and_mitigation",
    "attack_detection_without_mitigation",
    "attack_detection_with_mitigation"
]

service_labels = [
    "Benign",
    "Attack · mitigation off",
    "Attack · mitigation on"
]

attempts = np.array(
    [
        int(
            service_by[c][
                "request_attempts"
            ]
        )
        for c in service_order
    ]
)

transport_success = np.array(
    [
        int(
            service_by[c][
                "successful_transport_requests"
            ]
        )
        for c in service_order
    ]
)

failed = np.array(
    [
        int(
            service_by[c][
                "failed_or_timed_out_requests"
            ]
        )
        for c in service_order
    ]
)

http200 = np.array(
    [
        int(
            service_by[c][
                "http_200"
            ]
        )
        for c in service_order
    ]
)

http429 = np.array(
    [
        int(
            service_by[c][
                "http_429"
            ]
        )
        for c in service_order
    ]
)

if not np.all(
    transport_success == attempts
):
    raise RuntimeError(
        "Figure 09 transport accounting mismatch"
    )

if not np.all(
    failed == 0
):
    raise RuntimeError(
        "Figure 09 unexpected transport failures"
    )

if not np.all(
    http200 + http429 == attempts
):
    raise RuntimeError(
        "Figure 09 HTTP accounting mismatch"
    )

pct200 = (
    http200
    / attempts
    * 100
)

pct429 = (
    http429
    / attempts
    * 100
)

fig, ax = plt.subplots(
    figsize=(10.0, 5.9)
)

y = np.arange(
    len(service_order)
)

ax.barh(
    y,
    pct200,
    height=0.48,
    label="HTTP 200"
)

ax.barh(
    y,
    pct429,
    left=pct200,
    height=0.48,
    label="HTTP 429"
)

ax.set_yticks(
    y,
    service_labels
)

ax.set_xlim(
    0,
    100
)

ax.xaxis.set_major_formatter(
    PercentFormatter(
        xmax=100
    )
)

ax.set_xlabel(
    "Share of attempted requests"
)

ax.set_title(
    "Controlled-Lab Live Mitigation and Service Outcomes",
    loc="left",
    pad=14
)

ax.grid(
    axis="x"
)

for yi, n200, n429, p200, p429 in zip(
    y,
    http200,
    http429,
    pct200,
    pct429
):
    if p200 >= 12:
        ax.text(
            p200 / 2,
            yi,
            f"HTTP 200\n{n200:,} ({p200:.1f}%)",
            ha="center",
            va="center",
            fontsize=8
        )
    else:
        ax.annotate(
            f"HTTP 200: {n200:,} ({p200:.1f}%)",
            xy=(p200, yi),
            xytext=(8, -17),
            textcoords="offset points",
            fontsize=7.8
        )

    if p429 >= 12:
        ax.text(
            p200 + p429 / 2,
            yi,
            f"HTTP 429\n{n429:,} ({p429:.1f}%)",
            ha="center",
            va="center",
            fontsize=8
        )

ax.legend(
    frameon=False,
    loc="lower right"
)

ax.invert_yaxis()

m = service_by[
    "attack_detection_with_mitigation"
]

post_requests = int(
    m[
        "post_enforcement_targeted_requests"
    ]
)

post_429 = int(
    m[
        "post_enforcement_http_429"
    ]
)

post_failures = int(
    m[
        "post_enforcement_failures"
    ]
)

decision_ms = (
    float(
        m[
            "decision_to_enforcement_mean_seconds"
        ]
    )
    * 1000
)

if post_requests != post_429:
    raise RuntimeError(
        "Figure 09 post-enforcement accounting mismatch"
    )

if post_failures != 0:
    raise RuntimeError(
        "Figure 09 post-enforcement failure mismatch"
    )

footer = (
    f"All {attempts.sum():,} requests were transport-successful. In the mitigation-on "
    f"attack condition, {post_429:,}/{post_requests:,} targeted requests after enforcement "
    f"completion returned HTTP 429, with zero post-enforcement failures; mean "
    f"decision-to-enforcement time was {decision_ms:.3f} ms. The 381 HTTP 200 responses "
    "occurred before enforcement completion. HTTP 429 is application-layer enforcement, "
    "not firewall or network-drop evidence."
)

finish(
    fig,
    EXPECTED_FIGURES[8],
    footer=footer,
    bottom=0.23,
    left=0.23
)

actual = sorted(
    path.name
    for path in FIGURES.glob(
        "figure_*.png"
    )
)

if actual != EXPECTED_FIGURES:
    raise RuntimeError(
        f"Unexpected final figure set: {actual}"
    )

for name in actual:
    path = FIGURES / name

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"Empty figure: {name}"
        )

print(
    "TABLE_FREEZE_VERIFIED",
    True
)

print(
    "EVIDENCE_MATRIX_VERIFIED",
    True
)

print(
    "FIGURE_COUNT",
    len(actual)
)

print(
    "FINAL_FIGURE_SET",
    actual == EXPECTED_FIGURES
)

for name in actual:
    print(
        name,
        sha256(
            FIGURES / name
        )
    )