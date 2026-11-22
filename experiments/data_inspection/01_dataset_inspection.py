from pathlib import Path

DATASET_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab\dataset"
)

# Check the root directory
if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_ROOT}")

print("DATASET DIRECTORY STRUCTURE")
print("=" * 60)

for folder in DATASET_ROOT.iterdir():
    print(folder.name, "[Directory]" if folder.is_dir() else "[File]")

# Find our two recordings recursively
healthy_files = list(DATASET_ROOT.rglob("97_Normal_0.mat"))
faulty_files = list(DATASET_ROOT.rglob("105_0.mat"))

print("\nHEALTHY RECORDING")
print(healthy_files)

print("\nINNER-RACE FAULT RECORDING")
print(faulty_files)

if len(healthy_files) != 1:
    raise RuntimeError(
        f"Expected one healthy recording, found {len(healthy_files)}"
    )

if len(faulty_files) != 1:
    raise RuntimeError(
        f"Expected one faulty recording, found {len(faulty_files)}"
    )