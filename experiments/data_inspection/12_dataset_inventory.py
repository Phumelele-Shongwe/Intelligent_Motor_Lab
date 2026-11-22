
from pathlib import Path

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATASET_ROOT = PROJECT_ROOT / "dataset"

categories = {
    "Healthy": DATASET_ROOT / "Normal",
    "Inner Race": DATASET_ROOT / "12k_Drive_End_Bearing_Fault_Data" / "IR",
    "Ball": DATASET_ROOT / "12k_Drive_End_Bearing_Fault_Data" / "B",
    "Outer Race": DATASET_ROOT / "12k_Drive_End_Bearing_Fault_Data" / "OR",
}

print("\nCWRU DATASET INVENTORY")
print("=" * 65)

for category, folder in categories.items():
    print(f"\n{category}")
    print("-" * 65)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue

    recordings = sorted(folder.rglob("*.mat"))
    print(f"Total MAT files: {len(recordings)}")

    for recording in recordings:
        print(f"  {recording.relative_to(DATASET_ROOT)}")