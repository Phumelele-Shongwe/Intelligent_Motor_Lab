from pathlib import Path
from scipy.io import loadmat, whosmat
import numpy as np

# Dataset root
DATASET_ROOT = Path(
    r"C:\Users\pkans\Documents\Masters\labs\Intelligent_Motor_Lab\dataset"
)

# Exact file locations confirmed by our discovery script
healthy_path = (
    DATASET_ROOT / "Normal" / "97_Normal_0.mat"
)

faulty_path = (
    DATASET_ROOT
    / "12k_Drive_End_Bearing_Fault_Data"
    / "IR" / "007" / "105_0.mat"
)


def inspect_recording(file_path):

    print("\n" + "=" * 65)
    print(f"FILE: {file_path.name}")
    print("=" * 65)

    # Inspect MATLAB variable metadata
    variables = whosmat(file_path)

    print("\nMATLAB VARIABLES:")

    for name, shape, dtype in variables:
        print(f"{name:25} Shape: {shape}  Type: {dtype}")

    # Load actual MATLAB contents
    data = loadmat(file_path)

    return data

# Load both recordings
healthy_data = inspect_recording(healthy_path)
faulty_data = inspect_recording(faulty_path)


def extract_drive_end_signal(data):

    # Find the MATLAB variable ending in DE_time
    candidates = [
        name for name in data.keys()
        if name.endswith("_DE_time")
    ]

    if len(candidates) != 1:
        raise ValueError(
            f"Expected one DE_time variable, found: {candidates}"
        )

    variable_name = candidates[0]

    # Extract and convert to a 1D NumPy array
    signal = np.asarray(
        data[variable_name],
        dtype=np.float64
    ).ravel()

    return variable_name, signal


healthy_name, healthy_signal = extract_drive_end_signal(
    healthy_data
)

faulty_name, faulty_signal = extract_drive_end_signal(
    faulty_data
)

print("\nEXTRACTED SIGNALS")
print("-" * 50)

print("Healthy variable:", healthy_name)
print("Healthy shape:", healthy_signal.shape)

print("Faulty variable:", faulty_name)
print("Faulty shape:", faulty_signal.shape)


def calculate_statistics(signal, condition):

    return {
        "Condition": condition,
        "Samples": len(signal),
        "Mean": np.mean(signal),
        "Std": np.std(signal),
        "Minimum": np.min(signal),
        "Maximum": np.max(signal),
        "RMS": np.sqrt(np.mean(signal ** 2))
    }


healthy_stats = calculate_statistics(
    healthy_signal, "Healthy"
)

faulty_stats = calculate_statistics(
    faulty_signal, "Inner-race fault"
)

for result in [healthy_stats, faulty_stats]:

    print("\n", result["Condition"])

    for key, value in result.items():
        print(f"{key}: {value}")


import matplotlib.pyplot as plt

N_SAMPLES = 2000

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Healthy bearing
axes[0].plot(
    healthy_signal[:N_SAMPLES],
    linewidth=0.8
)

axes[0].set_title("Healthy Bearing - 97_Normal_0")
axes[0].set_xlabel("Sample index")
axes[0].set_ylabel("Acceleration")
axes[0].grid(True, alpha=0.3)

# Inner-race fault
axes[1].plot(
    faulty_signal[:N_SAMPLES],
    linewidth=0.8
)

axes[1].set_title("Inner-Race Fault - 105_0")
axes[1].set_xlabel("Sample index")
axes[1].set_ylabel("Acceleration")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Extract rotational speed metadata
healthy_rpm = float(healthy_data["X097RPM"].squeeze())
faulty_rpm = float(faulty_data["X105RPM"].squeeze())

print("\nMOTOR OPERATING CONDITIONS")
print("-" * 45)

print(f"Healthy RPM: {healthy_rpm}")
print(f"Faulty RPM: {faulty_rpm}")

# Convert RPM into rotational frequency
healthy_frequency = healthy_rpm / 60
faulty_frequency = faulty_rpm / 60

print(f"\nHealthy rotational frequency: {healthy_frequency:.2f} Hz")
print(f"Faulty rotational frequency: {faulty_frequency:.2f} Hz")