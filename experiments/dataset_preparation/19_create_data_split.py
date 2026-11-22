
from pathlib import Path
import pandas as pd

# ============================================================
# RECORDING-LEVEL TRAIN / TEST SPLIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "dataset"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "recording_split.csv"

# Explicitly reserve these recordings for independent testing.
TEST_RECORDINGS = {
    "100_Normal_3.mat",
    "105_0.mat",
    "170_1.mat",
    "211_2.mat",
    "3004_3.mat",
}

records = []

# ------------------------------------------------------------
# 1. Discover healthy recordings
# ------------------------------------------------------------

healthy_dir = DATASET_ROOT / "Normal"

for file in sorted(healthy_dir.glob("*.mat")):
    records.append({
        "filename": file.name,
        "filepath": str(file.relative_to(PROJECT_ROOT)),
        "fault_type": "Healthy",
        "label": 0,
    })

# ------------------------------------------------------------
# 2. Discover Inner Race recordings
# ------------------------------------------------------------

ir_dir = DATASET_ROOT / "12k_Drive_End_Bearing_Fault_Data" / "IR"

for file in sorted(ir_dir.rglob("*.mat")):
    records.append({
        "filename": file.name,
        "filepath": str(file.relative_to(PROJECT_ROOT)),
        "fault_type": "Inner_Race",
        "label": 1,
    })

df = pd.DataFrame(records)

# ------------------------------------------------------------
# 3. Assign recording-level split
# ------------------------------------------------------------

df["split"] = df["filename"].apply(
    lambda name: "test" if name in TEST_RECORDINGS else "train"
)

# ------------------------------------------------------------
# 4. Validate the allocation
# ------------------------------------------------------------

print("\nRECORDING-LEVEL DATA SPLIT")
print("=" * 65)

print(f"Total recordings: {len(df)}")
print(f"Training recordings: {(df['split'] == 'train').sum()}")
print(f"Testing recordings: {(df['split'] == 'test').sum()}")

print("\nCLASS DISTRIBUTION")
print(
    pd.crosstab(
        df["split"],
        df["fault_type"],
        margins=True
    )
)

# Verify exact recording counts.
assert len(df) == 20, "Expected exactly 20 recordings."
assert (df["split"] == "train").sum() == 15
assert (df["split"] == "test").sum() == 5

# Verify class counts.
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

assert (train["fault_type"] == "Healthy").sum() == 3
assert (train["fault_type"] == "Inner_Race").sum() == 12

assert (test["fault_type"] == "Healthy").sum() == 1
assert (test["fault_type"] == "Inner_Race").sum() == 4

# Verify the reserved recordings exist exactly once.
assert set(test["filename"]) == TEST_RECORDINGS

# Verify no recording overlap.
assert set(train["filename"]).isdisjoint(set(test["filename"]))

# ------------------------------------------------------------
# 5. Save manifest
# ------------------------------------------------------------

df = df.sort_values(["split", "fault_type", "filename"])

df.to_csv(OUTPUT_FILE, index=False)

print("\nTRAINING RECORDINGS")
print("-" * 65)

for _, row in train.sort_values("filename").iterrows():
    print(f"{row['fault_type']:12s} | {row['filename']}")

print("\nHELD-OUT TEST RECORDINGS")
print("-" * 65)

for _, row in test.sort_values("filename").iterrows():
    print(f"{row['fault_type']:12s} | {row['filename']}")

print("\nSUCCESS: No recording leakage detected.")
print(f"Manifest saved to: {OUTPUT_FILE}")