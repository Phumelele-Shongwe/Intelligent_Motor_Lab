
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

CLASSIFICATION_FILE = (
    METRICS_DIR / "existing_models_holdout_metrics.csv"
)
INFERENCE_FILE = METRICS_DIR / "inference_benchmark.csv"

OUTPUT_CSV = METRICS_DIR / "final_model_comparison.csv"
OUTPUT_REPORT = RESULTS_DIR / "final_experimental_comparison.md"
OUTPUT_FIGURE = FIGURES_DIR / "final_model_comparison.png"

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "cnn",
]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "cnn": "1D CNN",
}

CLASSIFICATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "roc_auc",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_model_name(value):
    value = str(value).strip().lower()

    aliases = {
        "logistic regression": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "random forest": "random_forest",
        "random_forest": "random_forest",
        "cnn": "cnn",
        "1d cnn": "cnn",
    }

    return aliases.get(value, value)


def file_size_kb(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")

    return path.stat().st_size / 1024


def markdown_table(dataframe):
    """Create a Markdown table without requiring tabulate."""

    headers = list(dataframe.columns)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in dataframe.iterrows():
        values = [
            str(value).replace("|", "\\|")
            for value in row.tolist()
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def validate_columns(dataframe, required, filename):
    missing = [column for column in required
               if column not in dataframe.columns]

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {missing}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

print("\nFINAL EXPERIMENTAL COMPARISON")
print("=" * 70)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("\nLOADING EXISTING RESULTS")
print("-" * 70)

classification = pd.read_csv(CLASSIFICATION_FILE)
inference = pd.read_csv(INFERENCE_FILE)

validate_columns(
    classification,
    ["model"] + CLASSIFICATION_METRICS,
    CLASSIFICATION_FILE.name,
)

validate_columns(
    inference,
    [
        "model",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "recordings_per_second",
    ],
    INFERENCE_FILE.name,
)

classification["model"] = (
    classification["model"].apply(normalize_model_name)
)

inference["model"] = (
    inference["model"].apply(normalize_model_name)
)

print("[OK] Classification metrics loaded")
print("[OK] Inference benchmark loaded")


# ============================================================
# MODEL STORAGE SIZES
# ============================================================

print("\nCALCULATING MODEL STORAGE")
print("-" * 70)

model_sizes = {
    "logistic_regression": file_size_kb(
        MODEL_DIR / "logistic_regression.joblib"
    ),
    "random_forest": file_size_kb(
        MODEL_DIR / "random_forest.joblib"
    ),
    "cnn": (
        file_size_kb(MODEL_DIR / "bearing_cnn.pth")
        + file_size_kb(MODEL_DIR / "cnn_normalization.json")
    ),
}

for model, size in model_sizes.items():
    print(f"{MODEL_LABELS[model]:22s} {size:.2f} KB")


# ============================================================
# MERGE RESULTS
# ============================================================

print("\nCONSOLIDATING RESULTS")
print("-" * 70)

comparison = classification.merge(
    inference,
    on="model",
    how="inner",
    validate="one_to_one",
)

if set(comparison["model"]) != set(MODEL_ORDER):
    raise ValueError(
        "The classification and inference results do not contain "
        "exactly the three expected models."
    )

comparison["model_size_kb"] = (
    comparison["model"].map(model_sizes)
)

comparison["model_label"] = (
    comparison["model"].map(MODEL_LABELS)
)

comparison = (
    comparison.set_index("model")
    .loc[MODEL_ORDER]
    .reset_index()
)

comparison.to_csv(OUTPUT_CSV, index=False)

print("[OK] Results merged")
print(f"[OK] Comparison CSV saved: {OUTPUT_CSV}")


# ============================================================
# CREATE FINAL COMPARISON FIGURE
# ============================================================

print("\nGENERATING FINAL COMPARISON FIGURE")
print("-" * 70)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.titlesize": 16,
})

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

fig.suptitle(
    "Intelligent Motor Laboratory: Model Comparison",
    fontweight="bold",
    y=0.98,
)

labels = comparison["model_label"].tolist()
x = np.arange(len(labels))


# ------------------------------------------------------------
# 1. CLASSIFICATION PERFORMANCE TABLE
# ------------------------------------------------------------

ax = axes[0, 0]
ax.axis("off")
ax.set_title(
    "Held-Out Classification Performance",
    fontweight="bold",
    pad=12,
)

metric_labels = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "recall_sensitivity": "Recall / Sensitivity",
    "specificity": "Specificity",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}

table_data = []

for metric in CLASSIFICATION_METRICS:
    values = [
        f"{value:.3f}"
        for value in comparison[metric]
    ]
    table_data.append([metric_labels[metric]] + values)

table = ax.table(
    cellText=table_data,
    colLabels=["Metric"] + labels,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.55)


# ------------------------------------------------------------
# 2. INFERENCE LATENCY
# ------------------------------------------------------------

ax = axes[0, 1]

bar_width = 0.35

ax.bar(
    x - bar_width / 2,
    comparison["mean_latency_ms"],
    bar_width,
    label="Mean latency",
)

ax.bar(
    x + bar_width / 2,
    comparison["p95_latency_ms"],
    bar_width,
    label="P95 latency",
)

ax.set_title("Inference Latency", fontweight="bold")
ax.set_ylabel("Milliseconds per recording")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale("log")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)


# ------------------------------------------------------------
# 3. INFERENCE THROUGHPUT
# ------------------------------------------------------------

ax = axes[1, 0]

bars = ax.bar(
    x,
    comparison["recordings_per_second"],
)

ax.set_title("Sequential Inference Throughput", fontweight="bold")
ax.set_ylabel("Recordings per second")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis="y", alpha=0.25)

for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{height:,.1f}",
        (bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )


# ------------------------------------------------------------
# 4. MODEL STORAGE
# ------------------------------------------------------------

ax = axes[1, 1]

bars = ax.bar(
    x,
    comparison["model_size_kb"],
)

ax.set_title("Saved Model Storage Size", fontweight="bold")
ax.set_ylabel("Kilobytes (KB)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis="y", alpha=0.25)

for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{height:.2f} KB",
        (bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )


fig.text(
    0.5,
    0.015,
    "CPU inference | One recording per call | "
    "CNN storage includes its normalization file",
    ha="center",
    fontsize=9,
)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])

fig.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"[OK] Figure saved: {OUTPUT_FIGURE}")


# ============================================================
# GENERATE EXPERIMENTAL REPORT
# ============================================================

print("\nGENERATING FINAL REPORT")
print("-" * 70)

display_columns = [
    "model_label",
    "accuracy",
    "precision",
    "recall_sensitivity",
    "f1",
    "roc_auc",
    "mean_latency_ms",
    "p95_latency_ms",
    "recordings_per_second",
    "model_size_kb",
]

report_table = comparison[display_columns].copy()

report_table.columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "Mean latency (ms)",
    "P95 latency (ms)",
    "Recordings/s",
    "Storage (KB)",
]

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
]:
    report_table[column] = report_table[column].map(
        lambda value: f"{value:.3f}"
    )

for column in [
    "Mean latency (ms)",
    "P95 latency (ms)",
]:
    report_table[column] = report_table[column].map(
        lambda value: f"{value:.4f}"
    )

report_table["Recordings/s"] = (
    report_table["Recordings/s"].map(
        lambda value: f"{value:.2f}"
    )
)

report_table["Storage (KB)"] = (
    report_table["Storage (KB)"].map(
        lambda value: f"{value:.2f}"
    )
)

metadata_path = MODEL_DIR / "holdout_experiment_metadata.json"

if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
else:
    metadata = {}

training_recordings = metadata.get("training_recordings", 15)
test_recordings = metadata.get("test_recordings", 5)
test_segments = metadata.get("test_segments", 75)

report = f"""# Intelligent Motor Laboratory
## Final Experimental Model Comparison

### 1. Experiment overview

This experiment compares three approaches to bearing fault
classification using vibration signals from the CWRU dataset:

- Logistic Regression using engineered statistical features.
- Random Forest using engineered statistical features.
- A one-dimensional Convolutional Neural Network (1D CNN)
  operating directly on raw vibration segments.

The experiment used {training_recordings} training recordings
and {test_recordings} held-out test recordings, containing
{test_segments} test segments.

### 2. Classification performance

{markdown_table(report_table[[
    "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
]])}

All three models achieved perfect classification metrics on
this particular held-out test set.

**Important limitation:** The test set contains only five
recordings, comprising one healthy recording and four faulty
recordings. These results are encouraging but do not establish
100% real-world diagnostic accuracy or generalization.

### 3. Computational efficiency

{markdown_table(report_table[[
    "Model", "Mean latency (ms)", "P95 latency (ms)",
    "Recordings/s"
]])}

Inference was measured on the CPU, with one recording processed
per call. Model loading and signal preprocessing were excluded
from the timed inference operation.

The CNN processes all available segments belonging to a recording
and averages their predicted fault probabilities. Traditional
models operate on one median-aggregated feature vector per
recording.

Throughput represents the reciprocal of mean sequential
per-recording latency. It is not a parallel-processing benchmark.

### 4. Model storage

{markdown_table(report_table[["Model", "Storage (KB)"]])}

The CNN storage figure includes both its saved checkpoint and
the normalization parameters required for inference.

### 5. Engineering interpretation

**Logistic Regression:** Provides the lowest inference latency
and smallest storage footprint. Its input is a compact set of
engineered features, making it computationally efficient.

**Random Forest:** Captures nonlinear relationships between
engineered features but has the highest mean inference latency
and largest saved model size in this experiment.

**1D CNN:** Learns representations directly from vibration
segments rather than relying on manually selected features.
It achieved the same observed classification metrics and had
lower mean inference latency than Random Forest in this benchmark.
Its inference latency is more variable and depends on the number
of segments processed per recording.

### 6. Overall conclusion

The experiment demonstrates three viable approaches to
bearing fault classification, each with different engineering
trade-offs involving feature representation, model complexity,
storage, and inference efficiency.

Logistic Regression is attractive where computational efficiency
is the priority. The CNN is particularly interesting where
learning directly from vibration signals is desirable. Random
Forest provides a nonlinear traditional machine-learning
baseline.

Further evaluation on a larger and more diverse set of
independent recordings is required before drawing conclusions
about real-world deployment performance.

---

### Generated artifacts

- `results/metrics/final_model_comparison.csv`
- `results/figures/final_model_comparison.png`
- `results/final_experimental_comparison.md`
"""

OUTPUT_REPORT.write_text(report, encoding="utf-8")

print(f"[OK] Report saved: {OUTPUT_REPORT}")


# ============================================================
# COMPLETION
# ============================================================

print("\nFINAL COMPARISON COMPLETED")
print("=" * 70)

print("\nGenerated files:")
print(f"1. {OUTPUT_CSV}")
print(f"2. {OUTPUT_FIGURE}")
print(f"3. {OUTPUT_REPORT}")

print("\nNo model training or model artifact modification occurred.")