
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATA_FILE = (
    PROJECT_ROOT
    / "Feature_Engineering"
    / "09_motor_features_binary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Feature_Engineering"
    / "feature_space_plots"
)

FEATURES = [
    "rms",
    "crest_factor",
    "envelope_dominant_frequency"
]

GROUP = "recording_id"
TARGET = "label"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD AND AGGREGATE DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

recording_data = (
    df.groupby(GROUP)
    .agg(
        rms=("rms", "median"),
        crest_factor=("crest_factor", "median"),
        envelope_dominant_frequency=(
            "envelope_dominant_frequency", "median"
        ),
        label=("label", "first"),
        fault_type=("fault_type", "first"),
    )
    .reset_index()
)

print("\nFEATURE-SPACE INVESTIGATION")
print("=" * 65)

print("Segment observations:", len(df))
print("Independent recordings:", len(recording_data))

print("\nRecording-level class distribution:")
print(recording_data["fault_type"].value_counts())

print("\nFEATURE SUMMARY BY CLASS")
print("=" * 65)

print(
    recording_data.groupby("fault_type")[FEATURES]
    .agg(["min", "median", "max"])
    .round(4)
)


# ============================================================
# 2. PAIRWISE FEATURE-SPACE VISUALISATION
# ============================================================

pairs = [
    ("rms", "crest_factor"),
    ("rms", "envelope_dominant_frequency"),
    ("crest_factor", "envelope_dominant_frequency")
]

fig, axes = plt.subplots(1, 3, figsize=(19, 5))

class_styles = {
    "Healthy": {"color": "tab:blue", "marker": "o"},
    "Inner_Race": {"color": "tab:red", "marker": "^"}
}

for ax, (x_feature, y_feature) in zip(axes, pairs):

    for fault_type, style in class_styles.items():

        subset = recording_data[
            recording_data["fault_type"] == fault_type
        ]

        ax.scatter(
            subset[x_feature],
            subset[y_feature],
            label=fault_type,
            color=style["color"],
            marker=style["marker"],
            s=85,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5
        )

    ax.set_xlabel(x_feature.replace("_", " ").title())
    ax.set_ylabel(y_feature.replace("_", " ").title())
    ax.set_title(f"{x_feature} vs {y_feature}")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(
    "Recording-Level Feature Space: Healthy vs Inner Race",
    fontsize=15
)

plt.tight_layout()

pairwise_path = OUTPUT_DIR / "12_pairwise_feature_space.png"
plt.savefig(pairwise_path, dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 3. INDIVIDUAL FEATURE DISTRIBUTIONS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

rng = np.random.default_rng(42)

for ax, feature in zip(axes, FEATURES):

    healthy = recording_data.loc[
        recording_data["label"] == 0, feature
    ].to_numpy()

    inner_race = recording_data.loc[
        recording_data["label"] == 1, feature
    ].to_numpy()

    data = [healthy, inner_race]

    ax.boxplot(
        data,
        tick_labels=["Healthy", "Inner Race"],
        showfliers=False
    )

    # Overlay individual recording observations.
    for position, values in enumerate(data, start=1):

        jitter = rng.normal(
            loc=position,
            scale=0.045,
            size=len(values)
        )

        ax.scatter(
            jitter,
            values,
            s=55,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5
        )

    ax.set_title(feature.replace("_", " ").title())
    ax.set_ylabel("Feature value")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Individual Feature Distributions by Bearing Condition",
    fontsize=15
)

plt.tight_layout()

distribution_path = (
    OUTPUT_DIR / "12_feature_distributions.png"
)

plt.savefig(distribution_path, dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 4. SAVE RECORDING-LEVEL DATA
# ============================================================

recording_data.to_csv(
    OUTPUT_DIR / "12_recording_level_features.csv",
    index=False
)

print("\nOUTPUT FILES")
print("=" * 65)
print("Pairwise plots:", pairwise_path)
print("Feature distributions:", distribution_path)
print(
    "Recording-level data:",
    OUTPUT_DIR / "12_recording_level_features.csv"
)

print("\nFEATURE-SPACE INVESTIGATION COMPLETE")