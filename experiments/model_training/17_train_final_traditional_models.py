
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "Feature_Engineering"
    / "09_motor_features_binary.csv"
)

SAVE_DIR = (
    PROJECT_ROOT
    / "Feature_Engineering"
    / "saved_models"
)

FEATURES = [
    "rms",
    "crest_factor",
    "envelope_dominant_frequency"
]

GROUP = "recording_id"
TARGET = "label"


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("\nFINAL TRADITIONAL MODEL TRAINING")
print("=" * 65)

df = pd.read_csv(DATA_FILE)

required_columns = FEATURES + [GROUP, TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError("Missing values detected in training data.")

# Each recording must have exactly one class label.
label_counts = df.groupby(GROUP)[TARGET].nunique()

if (label_counts != 1).any():
    raise ValueError(
        "Inconsistent labels detected within recordings."
    )

# Aggregate segment-level features into recording-level features.
recording_data = (
    df.groupby(GROUP)
    .agg(
        rms=("rms", "median"),
        crest_factor=("crest_factor", "median"),
        envelope_dominant_frequency=(
            "envelope_dominant_frequency", "median"
        ),
        label=("label", "first")
    )
    .reset_index()
)

X = recording_data[FEATURES]
y = recording_data[TARGET].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Expected binary labels: Healthy=0, Inner_Race=1."
    )

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Non-finite feature values detected.")

print("Segment observations:", len(df))
print("Independent recordings:", len(recording_data))
print("Features:", FEATURES)

print("\nRecording-level class distribution:")
print(y.value_counts().sort_index())


# ============================================================
# 3. DEFINE FINAL MODELS
# ============================================================

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=1.0
        ))
    ]),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
}


# ============================================================
# 4. TRAIN AND SAVE MODELS
# ============================================================

SAVE_DIR.mkdir(parents=True, exist_ok=True)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X, y)

    model_path = SAVE_DIR / f"{name}.joblib"

    joblib.dump(model, model_path)

    print("Saved:", model_path)

    # Training-set predictions are only a save/load sanity check.
    probabilities = model.predict_proba(X)[:, 1]

    print(
        "Training probability range:",
        f"{probabilities.min():.4f} - "
        f"{probabilities.max():.4f}"
    )


# ============================================================
# 5. SAVE MODEL METADATA
# ============================================================

metadata = {
    "task": "Binary bearing fault classification",
    "models": list(models.keys()),
    "features": FEATURES,
    "aggregation": "Median across segments per recording",
    "target": TARGET,
    "class_mapping": {
        "Healthy": 0,
        "Inner_Race": 1
    },
    "probability_output": "Probability of Inner_Race",
    "training_recordings": len(recording_data),
    "training_segments": len(df),
    "sampling_frequency": 12000
}

metadata_path = SAVE_DIR / "traditional_model_metadata.json"

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)

print("\nSaved metadata:", metadata_path)


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print("\nMODEL SAVING COMPLETE")
print("=" * 65)

for name in models:
    print(f"{name}: {SAVE_DIR / (name + '.joblib')}")

print("\nTraditional models are ready for inference.")