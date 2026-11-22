
from pathlib import Path
import re
import csv

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATASET_ROOT = PROJECT_ROOT / "dataset"

FAULT_ROOT = (
    DATASET_ROOT /
    "12k_Drive_End_Bearing_Fault_Data"
)

records = []

# --------------------------------------------------
# 1. Healthy recordings
# --------------------------------------------------

healthy_folder = DATASET_ROOT / "Normal"

for file in sorted(healthy_folder.rglob("*.mat")):

    load_match = re.search(r"_(\d+)$", file.stem)

    records.append({
        "fault_type": "Healthy",
        "fault_diameter_in": None,
        "outer_race_position": None,
        "load_hp": int(load_match.group(1)) if load_match else None,
        "filename": file.name,
        "relative_path": str(file.relative_to(DATASET_ROOT))
    })


# --------------------------------------------------
# 2. Fault recordings
# --------------------------------------------------

for fault_type in ["IR", "B", "OR"]:

    fault_folder = FAULT_ROOT / fault_type

    for file in sorted(fault_folder.rglob("*.mat")):

        # Fault diameter comes from the parent
        # severity folder: 007, 014, 021, etc.
        diameter_folder = file.relative_to(fault_folder).parts[0]

        diameter = int(diameter_folder) / 1000

        # Motor load is the final suffix.
        load_match = re.search(r"_(\d+)$", file.stem)

        load_hp = (
            int(load_match.group(1))
            if load_match else None
        )

        # Outer-race position may appear in either
        # a folder name or the filename.
        position_match = re.search(
            r"@(\d+)",
            str(file.relative_to(fault_folder))
        )

        position = (
            int(position_match.group(1))
            if position_match else None
        )

        records.append({
            "fault_type": fault_type,
            "fault_diameter_in": diameter,
            "outer_race_position": position,
            "load_hp": load_hp,
            "filename": file.name,
            "relative_path": str(file.relative_to(DATASET_ROOT))
        })


# --------------------------------------------------
# 3. Display and save inventory
# --------------------------------------------------

output_file = PROJECT_ROOT / "Inspect_Data" / "recording_metadata.csv"

fieldnames = [
    "fault_type",
    "fault_diameter_in",
    "outer_race_position",
    "load_hp",
    "filename",
    "relative_path"
]

with open(output_file, "w", newline="") as csvfile:

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(records)


print("\nRECORDING METADATA SUMMARY")
print("=" * 65)

print(f"Total recordings: {len(records)}")

for fault_type in ["Healthy", "IR", "B", "OR"]:

    subset = [
        r for r in records
        if r["fault_type"] == fault_type
    ]

    print(f"{fault_type:12}: {len(subset)} recordings")

print("\nMetadata saved to:")
print(output_file)