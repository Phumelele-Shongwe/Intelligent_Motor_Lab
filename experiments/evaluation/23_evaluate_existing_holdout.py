
import json
import sys
import joblib
import numpy as np
import pandas as pd
import torch

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make the project root available for importing src.
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import BearingCNN


RESULTS_DIR = PROJECT_ROOT / "results"
FEATURE_DIR = RESULTS_DIR / "prepared_data"

MODEL_DIR = PROJECT_ROOT / "models"

METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "rms",
    "crest_factor",
    "envelope_dominant_frequency"
]

TARGET = "label"
GROUP = "filename"

DEVICE = torch.device("cpu")
THRESHOLD = 0.5


# ============================================================
# 2. LOAD EXISTING DATA AND MODEL ARTIFACTS
# ============================================================

print("\nEVALUATION OF EXISTING HOLDOUT MODELS")
print("=" * 70)

print("\nThis script performs inference and evaluation only.")
print("No model training or fitting will be performed.")

train_features = pd.read_csv(
    FEATURE_DIR / "train_features.csv"
)

test_features = pd.read_csv(
    FEATURE_DIR / "test_features.csv"
)

test_data = np.load(
    FEATURE_DIR / "test_segments.npz"
)

X_test_cnn = test_data["X"].astype(np.float32)
y_test_cnn = test_data["y"].astype(np.int64)

test_recordings_cnn = np.asarray(
    test_data["recordings"]
).astype(str)


# ============================================================
# 3. VALIDATE THE HOLDOUT DATA
# ============================================================

required_columns = FEATURES + [GROUP, TARGET]

if not set(required_columns).issubset(test_features.columns):
    raise ValueError("Test feature file is missing required columns.")

if test_features[required_columns].isnull().any().any():
    raise ValueError("Missing values found in test features.")

if len(X_test_cnn) != len(test_features):
    raise ValueError(
        "CNN segments and traditional feature rows are not aligned."
    )

if len(y_test_cnn) != len(test_features):
    raise ValueError(
        "CNN labels and traditional feature rows are not aligned."
    )

if not np.array_equal(
    test_recordings_cnn,
    test_features[GROUP].astype(str).to_numpy()
):
    raise ValueError(
        "CNN recording identifiers do not match feature recording identifiers."
    )

if not np.array_equal(
    y_test_cnn,
    test_features[TARGET].to_numpy(dtype=np.int64)
):
    raise ValueError(
        "CNN labels do not match traditional feature labels."
    )

train_recording_names = set(
    train_features[GROUP].astype(str).unique()
)

test_recording_names = set(
    test_features[GROUP].astype(str).unique()
)

if not train_recording_names.isdisjoint(test_recording_names):
    raise ValueError("Recording leakage detected!")

if len(test_recording_names) != 5:
    raise ValueError(
        f"Expected 5 test recordings, found {len(test_recording_names)}."
    )

print("\nDATA VALIDATION")
print("-" * 70)
print("Training recordings:", len(train_recording_names))
print("Testing recordings:", len(test_recording_names))
print("Testing segments:", len(X_test_cnn))
print("CNN test shape:", X_test_cnn.shape)

print("\nTest recording class distribution:")

print(
    test_features.groupby(GROUP)[TARGET]
    .first()
    .value_counts()
    .sort_index()
    .rename(index={0: "Healthy", 1: "Inner_Race"})
)


# ============================================================
# 4. AGGREGATE TRADITIONAL FEATURES BY RECORDING
# ============================================================

def aggregate_recording_features(df):

    label_counts = df.groupby(GROUP)[TARGET].nunique()

    if (label_counts != 1).any():
        raise ValueError("Inconsistent labels within a recording.")

    recording_df = (
        df.groupby(GROUP)
        .agg(
            rms=("rms", "median"),
            crest_factor=("crest_factor", "median"),
            envelope_dominant_frequency=(
                "envelope_dominant_frequency",
                "median"
            ),
            label=("label", "first")
        )
        .reset_index()
    )

    X = recording_df[FEATURES].to_numpy(dtype=np.float64)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite traditional features.")

    return recording_df, X


test_recordings, X_test_trad = aggregate_recording_features(
    test_features
)

if len(test_recordings) != 5:
    raise ValueError("Expected five recording-level test observations.")


# ============================================================
# 5. LOAD EXISTING TRADITIONAL MODELS
# ============================================================

print("\nLOADING EXISTING TRADITIONAL MODELS")
print("-" * 70)

traditional_models = {}

for name in ["logistic_regression", "random_forest"]:

    model_path = MODEL_DIR / f"{name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    traditional_models[name] = joblib.load(model_path)

    print(f"[OK] {name} loaded")


# ============================================================
# 6. EVALUATE TRADITIONAL MODELS
# ============================================================

all_predictions = []

print("\nTRADITIONAL MODEL INFERENCE")
print("-" * 70)

for name, model in traditional_models.items():

    probabilities_all = model.predict_proba(X_test_trad)

    # Explicitly locate the probability column for class 1.
    positive_class_index = list(model.classes_).index(1)

    probabilities = probabilities_all[:, positive_class_index]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    for i, row in test_recordings.iterrows():

        all_predictions.append({
            "model": name,
            "filename": row[GROUP],
            "actual_class": int(row[TARGET]),
            "predicted_class": int(predictions[i]),
            "probability_inner_race": float(probabilities[i])
        })

    print(f"[OK] {name} inference completed")


# ============================================================
# 7. LOAD EXISTING CNN AND NORMALIZATION
# ============================================================

print("\nLOADING EXISTING CNN")
print("-" * 70)

cnn_model_path = MODEL_DIR / "bearing_cnn.pth"
normalization_path = MODEL_DIR / "cnn_normalization.json"

if not cnn_model_path.exists():
    raise FileNotFoundError(cnn_model_path)

if not normalization_path.exists():
    raise FileNotFoundError(normalization_path)

with open(normalization_path, "r") as file:
    normalization = json.load(file)

train_mean = float(normalization["mean"])
train_std = float(normalization["std"])

if not np.isfinite(train_mean) or not np.isfinite(train_std):
    raise ValueError("Invalid CNN normalization parameters.")

if train_std < 1e-8:
    raise ValueError("CNN normalization standard deviation is too small.")

checkpoint = torch.load(
    cnn_model_path,
    map_location=DEVICE,
    weights_only=True
)

cnn_model = BearingCNN().to(DEVICE)

cnn_model.load_state_dict(
    checkpoint["model_state_dict"]
)

cnn_model.eval()

print("[OK] CNN checkpoint loaded")
print("[OK] CNN architecture restored")
print("[OK] Training normalization loaded")
print("Training mean:", train_mean)
print("Training std:", train_std)


# ============================================================
# 8. CNN HELD-OUT INFERENCE
# ============================================================

print("\nCNN INFERENCE")
print("-" * 70)

# Use the original training-set normalization.
X_test_normalized = (
    X_test_cnn - train_mean
) / train_std

X_test_tensor = torch.tensor(
    X_test_normalized[:, np.newaxis, :],
    dtype=torch.float32
)

with torch.no_grad():

    logits = cnn_model(X_test_tensor)

    segment_probabilities = (
        torch.sigmoid(logits)
        .cpu()
        .numpy()
        .reshape(-1)
    )

cnn_results = pd.DataFrame({
    "filename": test_recordings_cnn,
    "actual_class": y_test_cnn,
    "segment_probability": segment_probabilities
})

# Average segment probabilities per independent recording.
cnn_recording_results = (
    cnn_results.groupby("filename")
    .agg(
        actual_class=("actual_class", "first"),
        probability_inner_race=("segment_probability", "mean")
    )
    .reset_index()
)

cnn_recording_results["predicted_class"] = (
    cnn_recording_results["probability_inner_race"] >= THRESHOLD
).astype(int)

if len(cnn_recording_results) != 5:
    raise ValueError("CNN did not produce five recording-level predictions.")

for _, row in cnn_recording_results.iterrows():

    all_predictions.append({
        "model": "cnn",
        "filename": row["filename"],
        "actual_class": int(row["actual_class"]),
        "predicted_class": int(row["predicted_class"]),
        "probability_inner_race": float(
            row["probability_inner_race"]
        )
    })

print("[OK] CNN segment inference completed")
print("[OK] Segment probabilities aggregated by recording")


# ============================================================
# 9. CALCULATE RECORDING-LEVEL TEST METRICS
# ============================================================

predictions_df = pd.DataFrame(all_predictions)

all_metrics = []

for name, group in predictions_df.groupby("model"):

    y_true = group["actual_class"].to_numpy()
    y_pred = group["predicted_class"].to_numpy()
    y_prob = group["probability_inner_race"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else np.nan
    )

    metrics = {
        "model": name,
        "test_recordings": len(group),

        "accuracy": accuracy_score(y_true, y_pred),

        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),

        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),

        "recall_sensitivity": recall_score(
            y_true, y_pred, zero_division=0
        ),

        "specificity": specificity,

        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),

        "roc_auc": roc_auc_score(y_true, y_prob),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

    all_metrics.append(metrics)


# ============================================================
# 10. SAVE EVALUATION RESULTS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)

predictions_path = (
    PREDICTIONS_DIR / "existing_models_holdout_predictions.csv"
)

metrics_path = (
    METRICS_DIR / "existing_models_holdout_metrics.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\nRECORDING-LEVEL PREDICTIONS")
print("=" * 70)

print(
    predictions_df.sort_values(
        ["model", "filename"]
    ).to_string(index=False)
)

print("\nMODEL PERFORMANCE")
print("=" * 70)

display_columns = [
    "model",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "tn",
    "fp",
    "fn",
    "tp"
]

print(
    metrics_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSAVED FILES")
print("-" * 70)
print("Predictions:", predictions_path)
print("Metrics:", metrics_path)

print("\nEVALUATION COMPLETED SUCCESSFULLY")
print("Existing model artifacts were not modified.")