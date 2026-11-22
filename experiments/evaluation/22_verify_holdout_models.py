
from pathlib import Path
import json
import joblib
import torch
import torch.nn as nn

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# ============================================================
# CNN ARCHITECTURE
# Must match the architecture used during training
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
# VERIFY ARTIFACTS
# ============================================================

print("\nHOLDOUT MODEL VERIFICATION")
print("=" * 65)

# 1. Logistic Regression
lr_path = MODEL_DIR / "logistic_regression.joblib"
lr_model = joblib.load(lr_path)

print("\n[OK] Logistic Regression loaded")
print("    Type:", type(lr_model).__name__)

# 2. Random Forest
rf_path = MODEL_DIR / "random_forest.joblib"
rf_model = joblib.load(rf_path)

print("\n[OK] Random Forest loaded")
print("    Type:", type(rf_model).__name__)

# 3. CNN checkpoint
cnn_path = MODEL_DIR / "bearing_cnn.pth"

checkpoint = torch.load(
    cnn_path,
    map_location="cpu",
    weights_only=True
)

if "model_state_dict" not in checkpoint:
    raise KeyError(
        "Expected 'model_state_dict' in CNN checkpoint."
    )

cnn_model = BearingCNN()
cnn_model.load_state_dict(checkpoint["model_state_dict"])
cnn_model.eval()

# Test a one-second input signal
dummy_input = torch.zeros(1, 1, 12000)

with torch.no_grad():
    output = cnn_model(dummy_input)

assert output.shape == (1,), (
    f"Unexpected CNN output shape: {output.shape}"
)

print("\n[OK] CNN loaded and inference test passed")
print("    Input shape:", tuple(dummy_input.shape))
print("    Output shape:", tuple(output.shape))

# 4. CNN normalization parameters
normalization_path = MODEL_DIR / "cnn_normalization.json"

with open(normalization_path, "r") as f:
    normalization = json.load(f)

print("\n[OK] CNN normalization parameters loaded")
print("    Mean:", normalization["mean"])
print("    Std:", normalization["std"])

# 5. Experiment metadata
metadata_path = METRICS_DIR / "holdout_experiment_metadata.json"

with open(metadata_path, "r") as f:
    metadata = json.load(f)

print("\n[OK] Experiment metadata loaded")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ALL MODEL ARTIFACTS VERIFIED SUCCESSFULLY")
print("=" * 65)