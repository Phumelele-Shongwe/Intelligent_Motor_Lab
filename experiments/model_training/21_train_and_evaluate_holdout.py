
import json
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    WeightedRandomSampler
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_DIR = RESULTS_DIR / "prepared_data"

SAVE_DIR = PROJECT_ROOT / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "rms",
    "crest_factor",
    "envelope_dominant_frequency"
]

TARGET = "label"
GROUP = "filename"

EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 0.001

FS = 12000
WINDOW_SAMPLES = 12000
EDGE_SAMPLES = 3000

DEVICE = torch.device("cpu")

SEED = 42


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 3. LOAD PREPARED DATA
# ============================================================

print("\nHELD-OUT BEARING CLASSIFICATION EXPERIMENT")
print("=" * 70)

train_features = pd.read_csv(
    FEATURE_DIR / "train_features.csv"
)

test_features = pd.read_csv(
    FEATURE_DIR / "test_features.csv"
)

train_data = np.load(
    FEATURE_DIR / "train_segments.npz"
)

test_data = np.load(
    FEATURE_DIR / "test_segments.npz"
)

X_train_cnn = train_data["X"].astype(np.float32)
y_train_cnn = train_data["y"].astype(np.float32)
train_recordings_cnn = train_data["recordings"]

X_test_cnn = test_data["X"].astype(np.float32)
y_test_cnn = test_data["y"].astype(np.int64)
test_recordings_cnn = test_data["recordings"]


# ============================================================
# 4. VALIDATE DATA SEPARATION
# ============================================================

train_recording_names = set(
    train_features[GROUP].unique()
)

test_recording_names = set(
    test_features[GROUP].unique()
)

assert train_recording_names.isdisjoint(
    test_recording_names
), "Recording leakage detected!"

assert set(train_recordings_cnn) == train_recording_names
assert set(test_recordings_cnn) == test_recording_names

assert len(train_recording_names) == 15
assert len(test_recording_names) == 5

assert len(X_train_cnn) == len(train_features)
assert len(X_test_cnn) == len(test_features)

print("\nDATA VALIDATION")
print("-" * 70)

print("Training recordings:", len(train_recording_names))
print("Testing recordings:", len(test_recording_names))

print("Training segments:", len(X_train_cnn))
print("Testing segments:", len(X_test_cnn))

print("CNN training shape:", X_train_cnn.shape)
print("CNN testing shape:", X_test_cnn.shape)

print("\nRecording-level class distribution:")

print(
    train_features.groupby(GROUP)["label"]
    .first()
    .value_counts()
    .sort_index()
    .rename(index={0: "Healthy", 1: "Inner_Race"})
)

print("\nTest recording-level class distribution:")

print(
    test_features.groupby(GROUP)["label"]
    .first()
    .value_counts()
    .sort_index()
    .rename(index={0: "Healthy", 1: "Inner_Race"})
)


# ============================================================
# 5. AGGREGATE TRADITIONAL FEATURES BY RECORDING
# ============================================================

def aggregate_recording_features(df):

    required = FEATURES + [GROUP, TARGET]

    if df[required].isnull().any().any():
        raise ValueError("Missing values in feature data.")

    label_counts = df.groupby(GROUP)[TARGET].nunique()

    if (label_counts != 1).any():
        raise ValueError("Inconsistent recording labels.")

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
    y = recording_df[TARGET].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite traditional features.")

    return recording_df, X, y


train_recordings, X_train_trad, y_train_trad = (
    aggregate_recording_features(train_features)
)

test_recordings, X_test_trad, y_test_trad = (
    aggregate_recording_features(test_features)
)

assert len(train_recordings) == 15
assert len(test_recordings) == 5


# ============================================================
# 6. DEFINE TRADITIONAL MODELS
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
# 7. TRAIN AND EVALUATE TRADITIONAL MODELS
# ============================================================

all_predictions = []
all_metrics = []

for name, model in models.items():

    print(f"\nTRAINING {name.upper()}")
    print("-" * 70)

    model.fit(X_train_trad, y_train_trad)

    # Save model separately from original experiment.
    model_path = SAVE_DIR / f"{name}.joblib"

    joblib.dump(model, model_path)

    # Independent test inference.
    probabilities = model.predict_proba(X_test_trad)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    for i, row in test_recordings.iterrows():

        all_predictions.append({
            "model": name,
            "filename": row[GROUP],
            "actual_class": int(row["label"]),
            "predicted_class": int(predictions[i]),
            "probability_inner_race": float(probabilities[i])
        })

    print("Saved:", model_path)


# ============================================================
# 8. CNN ARCHITECTURE
# ============================================================

class BearingCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# 9. PREPARE CNN TRAINING DATA
# ============================================================

set_seed(SEED)

print("\nCNN TRAINING")
print("=" * 70)

# Calculate normalization exclusively from training segments.
train_mean = float(X_train_cnn.mean())
train_std = float(X_train_cnn.std())

if train_std < 1e-8:
    raise ValueError("Training signal has near-zero variance.")

X_train_normalized = (
    X_train_cnn - train_mean
) / train_std

# Do not calculate normalization from test data.
X_test_normalized = (
    X_test_cnn - train_mean
) / train_std

X_train_tensor = torch.tensor(
    X_train_normalized[:, np.newaxis, :],
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_cnn,
    dtype=torch.float32
)

# Recording-balanced and class-balanced sampling.
train_segment_counts = pd.Series(
    train_recordings_cnn
).value_counts()

recording_labels = (
    train_features.groupby(GROUP)["label"]
    .first()
    .to_dict()
)

class_recording_counts = {
    label: sum(
        recording_labels[name] == label
        for name in train_segment_counts.index
    )
    for label in [0, 1]
}

sample_weights = []

for recording_name in train_recordings_cnn:

    label = recording_labels[recording_name]

    n_segments = train_segment_counts[recording_name]

    n_class_recordings = class_recording_counts[label]

    weight = 1.0 / (
        n_segments * n_class_recordings
    )

    sample_weights.append(weight)

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)


# ============================================================
# 10. TRAIN CNN
# ============================================================

model = BearingCNN().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

model.train()

for epoch in range(EPOCHS):

    epoch_loss = 0.0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch + 1:2d}/{EPOCHS} "
        f"Loss: {epoch_loss / len(loader):.4f}"
    )


# ============================================================
# 11. SAVE CNN AND NORMALIZATION
# ============================================================

CNN_MODEL_PATH = SAVE_DIR / "bearing_cnn.pth"

NORMALIZATION_PATH = SAVE_DIR / "cnn_normalization.json"

checkpoint = {

    "model_state_dict": model.state_dict(),

    "architecture": {
        "input_channels": 1,
        "conv_channels": [16, 32, 64],
        "kernel_sizes": [15, 9, 7],
        "dropout": 0.2,
        "output_classes": 1
    },

    "sampling_frequency": FS,
    "window_samples": WINDOW_SAMPLES,
    "edge_samples": EDGE_SAMPLES,

    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,

    "class_mapping": {
        "Healthy": 0,
        "Inner_Race": 1
    },

    "training_recordings": 15,
    "training_segments": len(X_train_cnn)
}

torch.save(checkpoint, CNN_MODEL_PATH)

normalization = {
    "mean": train_mean,
    "std": train_std,
    "sampling_frequency": FS,
    "window_samples": WINDOW_SAMPLES,
    "edge_samples": EDGE_SAMPLES
}

with open(NORMALIZATION_PATH, "w") as file:

    json.dump(normalization, file, indent=4)

print("\nCNN saved:", CNN_MODEL_PATH)
print("Normalization saved:", NORMALIZATION_PATH)
print("Training mean:", train_mean)
print("Training std:", train_std)


# ============================================================
# 12. CNN HELD-OUT INFERENCE
# ============================================================

model.eval()

X_test_tensor = torch.tensor(
    X_test_normalized[:, np.newaxis, :],
    dtype=torch.float32
)

with torch.no_grad():

    logits = model(X_test_tensor)

    segment_probabilities = torch.sigmoid(logits).numpy()

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
    cnn_recording_results["probability_inner_race"] >= 0.5
).astype(int)

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


# ============================================================
# 13. CALCULATE RECORDING-LEVEL TEST METRICS
# ============================================================

predictions_df = pd.DataFrame(all_predictions)

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
# 14. SAVE TEST RESULTS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)

predictions_path = (
    PREDICTIONS_DIR / "holdout_test_predictions.csv"
)

metrics_path = (
    METRICS_DIR / "holdout_test_metrics.csv"
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
# 15. SAVE EXPERIMENT METADATA
# ============================================================

metadata = {

    "experiment": "Recording-level held-out evaluation",

    "training_recordings": 15,

    "test_recordings": 5,

    "training_segments": len(X_train_cnn),

    "test_segments": len(X_test_cnn),

    "features": FEATURES,

    "traditional_aggregation": (
        "Median of segment features per recording"
    ),

    "cnn_aggregation": (
        "Mean of segment probabilities per recording"
    ),

    "cnn_normalization": (
        "Mean and standard deviation calculated "
        "from training segments only"
    ),

    "classification_threshold": 0.5,

    "random_seed": SEED,

    "models": [
        "logistic_regression",
        "random_forest",
        "cnn"
    ],

    "test_set_used_for_training": False
}

with open(
    METRICS_DIR / "holdout_experiment_metadata.json",
    "w"
) as file:

    json.dump(metadata, file, indent=4)


# ============================================================
# 16. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("HELD-OUT TEST RESULTS")
print("=" * 70)

print(
    metrics_df.round(4).to_string(index=False)
)

print("\nRECORDING-LEVEL PREDICTIONS")
print("-" * 70)

print(
    predictions_df.round(4).to_string(index=False)
)

print("\nOUTPUT FILES")
print("-" * 70)

print("Models:", SAVE_DIR)

print("Predictions:", predictions_path)

print("Metrics:", metrics_path)

print("\nHELD-OUT EXPERIMENT COMPLETE")