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
# 5. TRAIN FINAL CNN USING ALL RECORDINGS
# ============================================================

set_seed(42)

print("\nFINAL CNN TRAINING")
print("=" * 65)

# Combine all segments from all 20 recordings.
X_list = []
y_list = []
sample_weights = []

healthy_recordings = sum(r["label"] == 0 for r in recordings)
ir_recordings = sum(r["label"] == 1 for r in recordings)

for recording in recordings:

    n_segments = len(recording["segments"])

    X_list.extend(recording["segments"])
    y_list.extend([recording["label"]] * n_segments)

    n_class_recordings = (
        healthy_recordings
        if recording["label"] == 0
        else ir_recordings
    )

    weight = 1.0 / (n_segments * n_class_recordings)

    sample_weights.extend([weight] * n_segments)

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.float32)

print("Training recordings:", len(recordings))
print("Training segments:", len(X))

# Calculate normalization parameters from all training data.
train_mean = float(X.mean())
train_std = float(X.std())

if train_std < 1e-8:
    raise ValueError("Training signal has near-zero variance")

X = (X - train_mean) / train_std

# Prepare PyTorch tensors.
X_tensor = torch.tensor(
    X[:, np.newaxis, :],
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y,
    dtype=torch.float32
)

# Preserve recording-balanced and class-balanced sampling.
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

dataset = TensorDataset(X_tensor, y_tensor)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

# Initialise final model.
model = BearingCNN().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Train.
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

model.eval()


# ============================================================
# 6. SAVE FINAL MODEL AND PREPROCESSING PARAMETERS
# ============================================================

import json

SAVE_DIR = OUTPUT_DIR / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SAVE_DIR / "bearing_cnn.pth"
NORMALIZATION_PATH = SAVE_DIR / "cnn_normalization.json"

# Save model weights and architecture information.
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
    }
}

torch.save(checkpoint, MODEL_PATH)

# Save signal normalization separately.
normalization = {
    "mean": train_mean,
    "std": train_std,
    "sampling_frequency": FS,
    "window_samples": WINDOW_SAMPLES,
    "edge_samples": EDGE_SAMPLES
}

with open(NORMALIZATION_PATH, "w") as file:
    json.dump(normalization, file, indent=4)

print("\nMODEL SAVING COMPLETE")
print("=" * 65)

print("CNN model:", MODEL_PATH)
print("Normalization:", NORMALIZATION_PATH)
print("Training mean:", train_mean)
print("Training std:", train_std)

print("\nFINAL CNN READY FOR INFERENCE")