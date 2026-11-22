
from pathlib import Path
import scipy.io
import numpy as np
import pandas as pd
import re

PROJECT_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab"
)

DATASET_ROOT = PROJECT_ROOT / "dataset"

metadata_file = (
    PROJECT_ROOT / "Inspect_Data" / "recording_metadata.csv"
)

metadata = pd.read_csv(metadata_file)

results = []

print("\nRECORDING CONSISTENCY AUDIT")
print("=" * 75)

for _, row in metadata.iterrows():

    file_path = DATASET_ROOT / row["relative_path"]

    mat = scipy.io.loadmat(file_path)

    # MATLAB metadata variables begin with "__"
    variables = [
        key for key in mat.keys()
        if not key.startswith("__")
    ]

    # Identify vibration signal variables
    signal_variables = [
        key for key in variables
        if key.endswith("_DE_time") or key.endswith("_FE_time")
    ]

    for variable in signal_variables:

        signal = np.asarray(mat[variable]).squeeze()

        if signal.ndim != 1 or signal.size == 0:
            continue

        # Look for the RPM variable
        rpm_variables = [
            key for key in variables
            if key.endswith("RPM")
        ]

        rpm = None

        if rpm_variables:
            rpm = float(
                np.asarray(mat[rpm_variables[0]]).squeeze()
            )

        # Standard CWRU 12k Drive End recordings
        # are generally sampled at 12,000 Hz.
        # Do not assume this for other datasets.
        sampling_frequency = 12000

        results.append({
            "fault_type": row["fault_type"],
            "diameter_in": row["fault_diameter_in"],
            "load_hp": row["load_hp"],
            "position": row["outer_race_position"],
            "filename": row["filename"],
            "signal_variable": variable,
            "number_of_samples": signal.size,
            "sampling_frequency_hz": sampling_frequency,
            "duration_seconds": signal.size / sampling_frequency,
            "rpm": rpm,
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "rms": float(np.sqrt(np.mean(signal ** 2)))
        })

    print(
        f"Processed: {row['filename']} | "
        f"Signal channels found: {len(signal_variables)}"
    )


audit = pd.DataFrame(results)

output_file = (
    PROJECT_ROOT / "Inspect_Data" / "recording_audit.csv"
)

audit.to_csv(output_file, index=False)

print("\nAUDIT SUMMARY")
print("=" * 75)

print("Total signal channels:", len(audit))

print("\nSampling frequency distribution:")
print(audit["sampling_frequency_hz"].value_counts())

print("\nRPM summary:")
print(audit["rpm"].describe())

print("\nSignal channels:")
print(audit["signal_variable"].value_counts())

print("\nAudit saved to:")
print(output_file)