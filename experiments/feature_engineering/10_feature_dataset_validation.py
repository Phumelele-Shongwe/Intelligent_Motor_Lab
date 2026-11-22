
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 1. LOAD FEATURE DATASET
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATA_FILE = (
    PROJECT_ROOT
    / "Feature_Engineering"
    / "09_motor_features_binary.csv"
)

df = pd.read_csv(DATA_FILE)

metadata = [
    "recording_id",
    "signal_variable",
    "segment",
    "fault_type",
    "label",
    "rpm",
]

feature_columns = [
    col for col in df.columns
    if col not in metadata
]

X = df[feature_columns]

# ============================================================
# 2. BASIC DATA QUALITY
# ============================================================

print("\nDATASET VALIDATION")
print("=" * 65)

print("Rows:", len(df))
print("Unique recordings:", df["recording_id"].nunique())
print("Features:", len(feature_columns))

print("\nMISSING VALUES")
print(df[feature_columns].isna().sum())

print("\nINFINITE VALUES")
print(
    np.isinf(X.select_dtypes(include=np.number))
    .sum()
)

# ============================================================
# 3. FEATURE VARIANCE
# ============================================================

print("\nFEATURE VARIANCE")
print("=" * 65)

variances = X.var()

print(variances.sort_values().to_string())

constant_features = variances[
    variances == 0
].index.tolist()

print("\nConstant features:", constant_features)

# ============================================================
# 4. RECORDING-LEVEL FEATURE SUMMARY
# ============================================================

print("\nRECORDING-LEVEL FEATURE MEANS")
print("=" * 65)

recording_summary = (
    df.groupby(
        ["fault_type", "recording_id"]
    )[feature_columns]
    .mean()
)

print(recording_summary.to_string())

# ============================================================
# 5. FEATURE RANGE ACROSS RECORDINGS
# ============================================================

print("\nRECORDING-LEVEL FEATURE RANGE BY CLASS")
print("=" * 65)

for feature in [
    "rms",
    "crest_factor",
    "kurtosis",
    "envelope_rms",
    "envelope_dominant_frequency",
]:

    print(f"\nFEATURE: {feature}")

    for fault_type, group in recording_summary.groupby(
        level="fault_type"
    ):

        values = group[feature]

        print(
            f"{fault_type}: "
            f"min={values.min():.6f}, "
            f"median={values.median():.6f}, "
            f"max={values.max():.6f}"
        )

# ============================================================
# 6. CHECK LABEL CONSISTENCY
# ============================================================

print("\nRECORDING LABEL CONSISTENCY")
print("=" * 65)

label_counts = (
    df.groupby("recording_id")["label"]
    .nunique()
)

print("Recordings with inconsistent labels:",
      (label_counts > 1).sum())

print("\nVALIDATION COMPLETE")