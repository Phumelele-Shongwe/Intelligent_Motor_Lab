
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "predictions"
    / "existing_models_holdout_predictions.csv"
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZES = {
    "Logistic Regression": 1.41,
    "Random Forest": 189.06,
    "1D CNN": 85.64
}

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "cnn": "1D CNN"
}


# ============================================================
# 2. LOAD AND VALIDATE PREDICTIONS
# ============================================================

predictions = pd.read_csv(PREDICTIONS_FILE)

required_columns = {
    "model",
    "filename",
    "actual_class",
    "predicted_class",
    "probability_inner_race"
}

if not required_columns.issubset(predictions.columns):
    raise ValueError("Prediction CSV is missing required columns.")

if predictions.isnull().any().any():
    raise ValueError("Missing values found in predictions.")

predictions["model_display"] = (
    predictions["model"].map(MODEL_NAMES)
)

if predictions["model_display"].isnull().any():
    raise ValueError("Unknown model found in prediction data.")

print("PREDICTION DATA LOADED")
print("-" * 60)
print("Rows:", len(predictions))
print("Models:", predictions["model"].nunique())
print("Recordings:", predictions["filename"].nunique())


# ============================================================
# 3. FAULT PROBABILITY COMPARISON
# ============================================================

probability_table = predictions.pivot(
    index="filename",
    columns="model_display",
    values="probability_inner_race"
)

model_order = [
    "Logistic Regression",
    "Random Forest",
    "1D CNN"
]

probability_table = probability_table[model_order]

ax = probability_table.plot(
    kind="bar",
    figsize=(11, 6),
    width=0.8
)

ax.set_title(
    "Inner-Race Fault Probability by Recording",
    fontsize=14,
    pad=15
)

ax.set_xlabel("Test Recording")
ax.set_ylabel("Predicted Inner-Race Probability")

ax.set_ylim(0, 1.08)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.axhline(
    y=0.5,
    linestyle="--",
    linewidth=1.2,
    label="Classification Threshold (0.5)"
)

ax.legend(
    title="Model",
    loc="upper left",
    bbox_to_anchor=(1.01, 1)
)

ax.tick_params(axis="x", rotation=25)

plt.tight_layout()

probability_path = (
    FIGURES_DIR / "model_fault_probability_comparison.png"
)

plt.savefig(
    probability_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. MODEL SIZE COMPARISON
# ============================================================

sizes = pd.Series(MODEL_SIZES)

ax = sizes.plot(
    kind="bar",
    figsize=(8, 5),
    width=0.55
)

ax.set_title(
    "Saved Model Size Comparison",
    fontsize=14,
    pad=15
)

ax.set_xlabel("Model")
ax.set_ylabel("Storage Size (KB)")

ax.tick_params(axis="x", rotation=15)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f KB",
        padding=4
    )

plt.tight_layout()

size_path = FIGURES_DIR / "model_size_comparison.png"

plt.savefig(
    size_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. SUMMARY
# ============================================================

print("\nFIGURES GENERATED")
print("-" * 60)

print("Fault probability:", probability_path)
print("Model size:", size_path)

print("\nCOMPARISON FIGURES COMPLETED SUCCESSFULLY")