# ===========================================================
# Importing required libraries
# ===========================================================

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import LeaveOneOut
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

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATASET_ROOT = PROJECT_ROOT / "dataset"
NORMAL_ROOT = DATASET_ROOT / "Normal"
IR_ROOT = (
    DATASET_ROOT
    / "12k_Drive_End_Bearing_Fault_Data"
    / "IR"
)

OUTPUT_DIR = PROJECT_ROOT / "Feature_Engineering"
OUTPUT_FILE = OUTPUT_DIR / "15_cnn_predictions.csv"

FS = 12000
WINDOW_SECONDS = 1.0
EDGE_SECONDS = 0.25

WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
EDGE_SAMPLES = int(FS * EDGE_SECONDS)

EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

print("\nCNN BEARING CLASSIFICATION")
print("=" * 65)
print("Device:", DEVICE)
print("Sampling frequency:", FS)
print("Samples per segment:", WINDOW_SAMPLES)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 3. LOAD SOURCE RECORDINGS
# ============================================================

recordings = []


def extract_segments(signal):

    signal = np.asarray(signal).squeeze().astype(np.float32)

    # Remove the same edge region used in feature extraction.
    start = EDGE_SAMPLES
    end = len(signal) - EDGE_SAMPLES

    signal = signal[start:end]

    segments = []

    for i in range(0, len(signal) - WINDOW_SAMPLES + 1,
                   WINDOW_SAMPLES):

        segment = signal[i:i + WINDOW_SAMPLES]

        if len(segment) == WINDOW_SAMPLES:
            segments.append(segment)

    return segments


# Healthy recordings: explicitly select the correct variables.
healthy_files = {
    "97_Normal_0.mat": "X097_DE_time",
    "98_Normal_1.mat": "X098_DE_time",
    "99_Normal_2.mat": "X099_DE_time",
    "100_Normal_3.mat": "X100_DE_time"
}

for filename, variable in healthy_files.items():

    filepath = NORMAL_ROOT / filename
    mat = loadmat(filepath)

    if variable not in mat:
        raise KeyError(
            f"Expected variable {variable} not found in {filepath}"
        )

    segments = extract_segments(mat[variable])

    recordings.append({
        "recording_id": f"Normal\\{filename}",
        "fault_type": "Healthy",
        "label": 0,
        "segments": segments
    })


# Inner-race recordings: discover DE signals recursively.
ir_files = sorted(IR_ROOT.rglob("*.mat"))

for filepath in ir_files:

    mat = loadmat(filepath)

    de_variables = [
        key for key in mat.keys()
        if key.endswith("_DE_time")
    ]

    if len(de_variables) != 1:
        raise ValueError(
            f"Expected exactly one DE signal in {filepath}; "
            f"found {de_variables}"
        )

    variable = de_variables[0]
    segments = extract_segments(mat[variable])

    recordings.append({
        "recording_id": str(filepath.relative_to(DATASET_ROOT)),
        "fault_type": "Inner_Race",
        "label": 1,
        "segments": segments
    })


# Validate the recording inventory.
if len(recordings) != 20:
    raise ValueError(
        f"Expected 20 recordings, found {len(recordings)}"
    )

for recording in recordings:
    if len(recording["segments"]) == 0:
        raise ValueError(
            f"No segments extracted: {recording['recording_id']}"
        )

print("\nRECORDING INVENTORY")
print("=" * 65)

for recording in recordings:
    print(
        f"{recording['fault_type']:12s} "
        f"{len(recording['segments']):3d} segments  "
        f"{recording['recording_id']}"
    )

print("\nTotal recordings:", len(recordings))
print(
    "Total segments:",
    sum(len(r["segments"]) for r in recordings)
)


# ============================================================
# 4. CNN ARCHITECTURE
# ============================================================

class BearingCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=15,
                padding=7
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=9,
                padding=4
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                padding=3
            ),
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
# 5. LEAVE-ONE-RECORDING-OUT CROSS-VALIDATION
# ============================================================

loo = LeaveOneOut()

y_true = []
y_pred = []
y_probability = []

for fold, (train_idx, test_idx) in enumerate(
    loo.split(np.arange(len(recordings))), start=1
):

    set_seed(42 + fold)

    train_recordings = [
        recordings[i] for i in train_idx
    ]

    test_recording = recordings[test_idx[0]]

    print("\nFOLD", fold, "/ 20")
    print("-" * 65)
    print("Held-out recording:", test_recording["recording_id"])

    # --------------------------------------------------------
    # Prepare training segments and their recording identities
    # --------------------------------------------------------

    X_train_list = []
    y_train_list = []
    recording_class_counts = {}

    for recording in train_recordings:

        n_segments = len(recording["segments"])

        recording_class_counts[
            recording["recording_id"]
        ] = (
            recording["label"],
            n_segments
        )

        for segment in recording["segments"]:

            X_train_list.append(segment)
            y_train_list.append(recording["label"])

    X_train = np.asarray(X_train_list, dtype=np.float32)
    y_train = np.asarray(y_train_list, dtype=np.float32)

    # --------------------------------------------------------
    # Training-only signal standardisation
    # --------------------------------------------------------

    train_mean = float(X_train.mean())
    train_std = float(X_train.std())

    if train_std < 1e-8:
        raise ValueError("Training signal has near-zero variance")

    X_train = (X_train - train_mean) / train_std

    X_test = np.asarray(
        test_recording["segments"],
        dtype=np.float32
    )

    X_test = (X_test - train_mean) / train_std

    # PyTorch input shape: (samples, channels, time)
    X_train_tensor = torch.tensor(
        X_train[:, np.newaxis, :],
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test[:, np.newaxis, :],
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Recording-balanced and class-balanced sampling
    # --------------------------------------------------------

    healthy_recordings = sum(
        r["label"] == 0 for r in train_recordings
    )

    ir_recordings = sum(
        r["label"] == 1 for r in train_recordings
    )

    sample_weights = []

    for recording in train_recordings:

        n_segments = len(recording["segments"])

        n_class_recordings = (
            healthy_recordings
            if recording["label"] == 0
            else ir_recordings
        )

        weight = 1.0 / (
            n_segments * n_class_recordings
        )

        sample_weights.extend(
            [weight] * n_segments
        )

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler
    )

    # --------------------------------------------------------
    # Initialise and train CNN
    # --------------------------------------------------------

    model = BearingCNN().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for epoch in range(EPOCHS):

        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 4 == 0:
            print(
                f"Epoch {epoch + 1:2d}/{EPOCHS} "
                f"Loss: {epoch_loss / len(train_loader):.4f}"
            )

    # --------------------------------------------------------
    # Evaluate all segments from held-out recording
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(X_test_tensor.to(DEVICE))

        segment_probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

    # One prediction per original recording.
    recording_probability = float(
        np.mean(segment_probabilities)
    )

    recording_prediction = int(
        recording_probability >= 0.5
    )

    y_true.append(test_recording["label"])
    y_pred.append(recording_prediction)
    y_probability.append(recording_probability)

    print("Held-out segments:", len(X_test))
    print(
        "Mean IR probability:",
        round(recording_probability, 4)
    )
    print("Predicted label:", recording_prediction)
    print("Actual label:", test_recording["label"])


# ============================================================
# 6. FINAL EVALUATION
# ============================================================

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)
y_probability = np.asarray(y_probability)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0 else 0
)

print("\nFINAL CNN RESULTS")
print("=" * 65)

print("Accuracy:", accuracy_score(y_true, y_pred))
print(
    "Balanced accuracy:",
    balanced_accuracy_score(y_true, y_pred)
)
print(
    "Precision:",
    precision_score(y_true, y_pred, zero_division=0)
)
print(
    "Sensitivity:",
    recall_score(y_true, y_pred, zero_division=0)
)
print("Specificity:", specificity)
print("F1-score:", f1_score(y_true, y_pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_true, y_probability))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


# ============================================================
# 7. SAVE RECORDING-LEVEL PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "recording_id": [
        recordings[i]["recording_id"]
        for i in range(len(recordings))
    ],
    "fault_type": [
        recordings[i]["fault_type"]
        for i in range(len(recordings))
    ],
    "true_label": y_true,
    "predicted_label": y_pred,
    "fault_probability": y_probability
})

results["correct"] = (
    results["true_label"] == results["predicted_label"]
)

results.to_csv(OUTPUT_FILE, index=False)

print("\nINDIVIDUAL RECORDING PREDICTIONS")
print("=" * 65)
print(results.to_string(index=False))

print("\nSaved:", OUTPUT_FILE)
print("\nCNN EXPERIMENT COMPLETE")